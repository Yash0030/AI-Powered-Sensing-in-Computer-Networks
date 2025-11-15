#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/energy-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/netanim-module.h" 
#include "ns3/error-model.h"     

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <map>
#include <queue>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace ns3;
using json = nlohmann::json;

// ============================================================================
// CONFIGURATION (MATCHES PYTHON CLIENT - STATE_DIM = 5)
// ============================================================================

const int STATE_DIM = 5;  // [edge_load, cloud_load, energy_level, task_size_MB, task_local_compute_s]
const int ACTION_DIM = 3; // 0: Local, 1: Edge, 2: Cloud
const int SIMULATION_STEPS_PER_EPISODE = 100;

struct TaskType {
    std::string name;
    double probability;
    uint32_t inputSizeBytes;
    double localProcessingTimeS;
    double localEnergyJ;
    double edgeSpeedup;
    double cloudSpeedup;
    bool isExtreme = false;
};

struct CarbonIntensity {
    std::string region;
    double gCO2_per_kWh;
    double renewablePercent;
    
    CarbonIntensity(const std::string& r, double co2, double renewable)
        : region(r), gCO2_per_kWh(co2), renewablePercent(renewable) {}
    
    CarbonIntensity() : region(""), gCO2_per_kWh(0.0), renewablePercent(0.0) {}
};

struct SimConfig {
    // Device Parameters
    int numUserDevices = 5;
    double deviceIdlePowerW = 0.5;
    double deviceTxPowerWiFiW = 1.5;
    double deviceRxPowerW = 0.8;
    double deviceComputeHeavyW = 0.3; // Local compute is cheap
    double batteryCapacityJ = 18000.0; 
    
    // Network Parameters
    std::string wifiStandard = "11ac";
    double wifiDataRateMbps = 100.0; 
    double wifiRange = 50.0;
    double wanDelayMeanMs = 300.0; // High cloud delay
    double wanDelayStdDevMs = 10.0;
    double wanBandwidthMbps = 100.0;
    double packetLossRate = 0.01; 
    
    // Server Parameters
    double edgeServerBasePowerW = 150.0;
    double edgeServerMaxPowerW = 400.0;
    int edgeServerCpuCores = 16;
    double edgeServerUsersSharing = 50.0;
    
    double cloudServerBasePowerW = 250.0;
    double cloudServerMaxPowerW = 600.0;
    int cloudServerCpuCores = 64;
    double cloudServerUsersSharing = 100.0;
    
    // Carbon Parameters
    std::string carbonRegion = "US-California";
    bool carbonTimeVariation = true;
    
    // Simulation
    int simulationStepsPerEpisode = SIMULATION_STEPS_PER_EPISODE;
    
    // Reward Weights: Delay penalty is reduced
    double rewardWeightDelay = 0.05; 
    double rewardWeightServerEnergy = 0.1;
    double rewardWeightCarbon = 0.1;
    
    std::vector<TaskType> taskTypes;
    std::string zmqPort = "5555";
};

// ============================================================================
// CARBON DATABASE
// ============================================================================

std::map<std::string, CarbonIntensity> CARBON_DATABASE = {
    {"US-California", CarbonIntensity("US-California", 200.0, 0.60)},
    {"US-Texas", CarbonIntensity("US-Texas", 450.0, 0.25)},
    {"Germany", CarbonIntensity("Germany", 350.0, 0.45)},
    {"France", CarbonIntensity("France", 60.0, 0.90)},
    {"Iceland", CarbonIntensity("Iceland", 10.0, 0.99)},
    {"India", CarbonIntensity("India", 650.0, 0.20)},
    {"China", CarbonIntensity("China", 550.0, 0.28)}
};

// ============================================================================
// REALISTIC BATTERY MODEL
// ============================================================================

class RealisticBatteryModel : public Object {
public:
    static TypeId GetTypeId() {
        static TypeId tid = TypeId("RealisticBatteryModel")
            .SetParent<Object>()
            .SetGroupName("Energy");
        return tid;
    }
    
    RealisticBatteryModel()
        : m_remainingCapacityJ(18000.0),
          m_initialCapacityJ(18000.0) {}
    
    void SetInitialEnergy(double energyJ) {
        m_initialCapacityJ = energyJ;
        m_remainingCapacityJ = energyJ;
    }
    
    double GetRemainingEnergy() const { return m_remainingCapacityJ; }
    
    void ConsumeEnergy(double energyJ) {
        m_remainingCapacityJ = std::max(0.0, m_remainingCapacityJ - energyJ);
    }
    
    bool IsDepleted() const { return m_remainingCapacityJ < 100.0; }
    
    double GetStateOfCharge() const {
        return m_remainingCapacityJ / m_initialCapacityJ;
    }
    
private:
    double m_remainingCapacityJ;
    double m_initialCapacityJ;
};

// ============================================================================
// THERMAL MODEL
// ============================================================================

class ThermalModel {
public:
    ThermalModel() : m_temperature(25.0) {}
    
    void UpdateTemperature(double powerW, double deltaTimeS) {
        double heatGenerated = powerW * deltaTimeS * 0.8;
        double heatDissipated = (m_temperature - 25.0) * 0.05 * deltaTimeS;
        m_temperature += (heatGenerated - heatDissipated) / 100.0;
        m_temperature = std::max(25.0, std::min(85.0, m_temperature));
    }
    
    double GetThrottleFactor() const {
        if (m_temperature < 70.0) return 1.0;
        if (m_temperature >= 85.0) return 0.5;
        return 1.0 - (0.5 * (m_temperature - 70.0) / 15.0);
    }
    
    double GetTemperature() const { return m_temperature; }
    
private:
    double m_temperature;
};

// ============================================================================
// SERVER MODEL
// ============================================================================

class ServerModel {
public:
    ServerModel(double basePowerW, double maxPowerW, int cores, double usersSharing)
        : m_basePowerW(basePowerW),
          m_maxPowerW(maxPowerW),
          m_totalCores(cores),
          m_availableCores(cores),
          m_usersSharing(usersSharing),
          m_currentUtilization(0.0) {}
    
    double ProcessTask(const TaskType& task, double speedup) {
        double baseProcessingTime = task.localProcessingTimeS / speedup;
        double queueingDelay = CalculateQueueingDelay();
        double totalDelay = baseProcessingTime + queueingDelay;
        
        double coresNeeded = 1.0;
        m_availableCores = std::max(0.0, m_availableCores - coresNeeded);
        
        Simulator::Schedule(Seconds(baseProcessingTime), 
                            &ServerModel::ReleaseCores, this, coresNeeded);
        
        m_currentUtilization = 1.0 - (m_availableCores / m_totalCores);
        return totalDelay;
    }
    
    double GetCurrentPower() const {
        double util = m_currentUtilization;
        return m_basePowerW + (m_maxPowerW - m_basePowerW) * std::pow(util, 3);
    }
    
    double GetPerUserPower() const {
        return GetCurrentPower() / m_usersSharing;
    }
    
    double GetUtilization() const { return m_currentUtilization; }
    
private:
    void ReleaseCores(double cores) {
        m_availableCores = std::min(m_totalCores, m_availableCores + cores);
        m_currentUtilization = 1.0 - (m_availableCores / m_totalCores);
    }
    
    double CalculateQueueingDelay() const {
        double rho = m_currentUtilization;
        if (rho >= 0.95) return 5.0; // High penalty for >95% utilization
        if (rho < 0.3) return 0.001;
        return 0.05 / (1.0 - rho); // M/M/1 queue approximation
    }
    
    double m_basePowerW;
    double m_maxPowerW;
    double m_totalCores;
    double m_availableCores;
    double m_usersSharing;
    double m_currentUtilization;
};

// ============================================================================
// CARBON CALCULATOR
// ============================================================================

class CarbonCalculator {
public:
    static double GetCurrentCarbonIntensity(const std::string& region, bool timeVariation) {
        if (CARBON_DATABASE.find(region) == CARBON_DATABASE.end()) {
            return CARBON_DATABASE["US-California"].gCO2_per_kWh;
        }
        
        double baseIntensity = CARBON_DATABASE[region].gCO2_per_kWh;
        
        if (!timeVariation) return baseIntensity;
        
        double hour = fmod(Simulator::Now().GetHours(), 24.0);
        double renewablePercent = CARBON_DATABASE[region].renewablePercent;
        
        double renewableFactor = 1.0;
        if (hour >= 10.0 && hour <= 16.0) { // Peak solar
            renewableFactor = 1.5;
        } else if (hour >= 0.0 && hour <= 6.0) { // Night (no solar)
            renewableFactor = 0.6;
        }
        
        double effectiveRenewable = std::min(1.0, renewablePercent * renewableFactor);
        double fossilFraction = 1.0 - effectiveRenewable;
        double baseFossilFraction = 1.0 - renewablePercent;
        
        if (baseFossilFraction < 0.01) return baseIntensity; // Avoid division by zero
        return baseIntensity * (fossilFraction / baseFossilFraction);
    }
    
    static double ConvertToPerJoule(double gCO2_per_kWh) {
        return gCO2_per_kWh / 3600000.0;
    }
};

// ============================================================================
// SIMULATION ENVIRONMENT
// ============================================================================

class SimulationEnv {
public:
    SimulationEnv(const SimConfig& config);
    json Reset();
    json Step(int action);
    json GetState();
    void SetupNetwork();
    void PrintComparisonReport();
    void PrintProgressReport();
    
private:
    TaskType SelectRandomTask();
    void UpdateStatistics(int action, double userEnergy, double serverEnergy, 
                          double delay, double carbon, double serverCarbon);
    
    SimConfig m_config;
    std::mt19937 m_rng;
    
    NodeContainer m_userDevices;
    NodeContainer m_edgeNode;
    NodeContainer m_cloudNode;
    
    std::vector<Ptr<RealisticBatteryModel>> m_batteries;
    std::vector<ThermalModel> m_thermalModels;
    
    ServerModel m_edgeServer;
    ServerModel m_cloudServer;
    
    NetDeviceContainer m_staDevices;
    NetDeviceContainer m_apDevice;
    Ipv4InterfaceContainer m_staInterfaces;
    Ipv4InterfaceContainer m_apInterface;
    Ipv4InterfaceContainer m_edgeCloudInterface;
    
    std::vector<Ptr<Application>> m_clientApps;
    
    int m_currentStep;
    int m_currentEpisode;
    TaskType m_currentTask; 
    
    // Statistics
    std::vector<double> m_totalUserEnergyPerAction;
    std::vector<double> m_totalServerEnergyPerAction;
    std::vector<double> m_totalDelayPerAction;
    std::vector<double> m_totalCarbonPerAction;
    std::vector<uint64_t> m_actionCounts;
    
    Ptr<FlowMonitor> m_flowMonitor;
    FlowMonitorHelper m_flowHelper;
};

// ============================================================================
// CONSTRUCTOR
// ============================================================================

SimulationEnv::SimulationEnv(const SimConfig& config)
    : m_config(config),
      m_rng(std::random_device{}()),
      m_edgeServer(config.edgeServerBasePowerW, config.edgeServerMaxPowerW,
                   config.edgeServerCpuCores, config.edgeServerUsersSharing),
      m_cloudServer(config.cloudServerBasePowerW, config.cloudServerMaxPowerW,
                    config.cloudServerCpuCores, config.cloudServerUsersSharing),
      m_currentStep(0),
      m_currentEpisode(0) {
    
    m_totalUserEnergyPerAction.assign(3, 0.0);
    m_totalServerEnergyPerAction.assign(3, 0.0);
    m_totalDelayPerAction.assign(3, 0.0);
    m_totalCarbonPerAction.assign(3, 0.0);
    m_actionCounts.assign(3, 0);
}

// ============================================================================
// NETWORK SETUP
// ============================================================================

void SimulationEnv::SetupNetwork() {
    m_userDevices = NodeContainer();
    m_edgeNode = NodeContainer();
    m_cloudNode = NodeContainer();
    m_batteries.clear();
    m_thermalModels.clear();
    m_clientApps.clear();
    
    m_userDevices.Create(m_config.numUserDevices);
    m_edgeNode.Create(1);
    m_cloudNode.Create(1);
    
    InternetStackHelper internet;
    internet.Install(m_userDevices);
    internet.Install(m_edgeNode);
    internet.Install(m_cloudNode);
    
    // WiFi Setup
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    
    WifiHelper wifi;
    if (m_config.wifiStandard == "11ac") {
         wifi.SetStandard(WIFI_STANDARD_80211ac);
    } else if (m_config.wifiStandard == "11n") {
         wifi.SetStandard(WIFI_STANDARD_80211n);
    } else {
         wifi.SetStandard(WIFI_STANDARD_80211g);
    }
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    WifiMacHelper wifiMac;
    Ssid ssid = Ssid("EdgeNetwork");
    
    wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    m_apDevice = wifi.Install(wifiPhy, wifiMac, m_edgeNode);
    
    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    m_staDevices = wifi.Install(wifiPhy, wifiMac, m_userDevices);
    
    // Mobility
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> edgePosition = CreateObject<ListPositionAllocator>();
    edgePosition->Add(Vector(0.0, 0.0, 0.0));
    mobility.SetPositionAllocator(edgePosition);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(m_edgeNode);
    
    mobility.SetPositionAllocator("ns3::RandomDiscPositionAllocator",
                                  "X", DoubleValue(0.0),
                                  "Y", DoubleValue(0.0),
                                  "Rho", StringValue("ns3::UniformRandomVariable[Min=0|Max=30]"));
    mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds", RectangleValue(Rectangle(-50, 50, -50, 50)));
    mobility.Install(m_userDevices);
    

    // WAN Link
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(m_config.wanBandwidthMbps * 1e6)));
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(m_config.wanDelayMeanMs)));
    NetDeviceContainer edgeCloudDevices = p2p.Install(m_edgeNode.Get(0), m_cloudNode.Get(0));
    
    // Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
    // em->SetAttribute("ErrorRate", DoubleValue(m_config.packetLossRate));
    // edgeCloudDevices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));

    // IP Addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    m_staInterfaces = address.Assign(m_staDevices);
    m_apInterface = address.Assign(m_apDevice);
    
    address.SetBase("10.1.2.0", "255.255.255.0");
    m_edgeCloudInterface = address.Assign(edgeCloudDevices);
    
    
    
   
    
    // Packet Sinks
    uint16_t port = 9000;
    PacketSinkHelper edgeSink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer edgeSinkApp = edgeSink.Install(m_edgeNode.Get(0));
    edgeSinkApp.Start(Seconds(0.0));
    edgeSinkApp.Stop(Hours(24.0));
    
    PacketSinkHelper cloudSink("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer cloudSinkApp = cloudSink.Install(m_cloudNode.Get(0));
    cloudSinkApp.Start(Seconds(0.0));
    cloudSinkApp.Stop(Hours(24.0));
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
     // Energy Models
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_config.numUserDevices); ++i) {
        Ptr<RealisticBatteryModel> battery = CreateObject<RealisticBatteryModel>();
        battery->SetInitialEnergy(m_config.batteryCapacityJ);
        m_batteries.push_back(battery);
        
        ThermalModel thermal;
        m_thermalModels.push_back(thermal);
    }
    // Install FlowMonitor
    m_flowMonitor = m_flowHelper.InstallAll();
}

// ============================================================================
// TASK SELECTION
// ============================================================================

TaskType SimulationEnv::SelectRandomTask() {
    if (m_config.taskTypes.empty()) {
        // Fallback task
        return {"default", 1.0, 100000, 0.5, 2.0, 3.0, 10.0, false};
    }
    
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double rand = dist(m_rng);
    double cumProb = 0.0;
    
    for (const auto& task : m_config.taskTypes) {
        cumProb += task.probability;
        if (rand <= cumProb) return task;
    }
    
    return m_config.taskTypes.back();
}

// ============================================================================
// GET STATE
// ============================================================================

json SimulationEnv::GetState() {
    double avgBatteryLevel = 0.0;
    for (const auto& battery : m_batteries) {
        avgBatteryLevel += battery->GetStateOfCharge();
    }
    avgBatteryLevel /= m_batteries.size();
    
    double edgeUtil = m_edgeServer.GetUtilization();
    double cloudUtil = m_cloudServer.GetUtilization();
    
    json state_array;
    state_array[0] = edgeUtil;      // Edge load
    state_array[1] = cloudUtil;     // Cloud load  
    state_array[2] = avgBatteryLevel; // Energy level
    
    // This helper function just returns the base 3-dim state
    return {{"state", state_array}};
}

// ============================================================================
// RESET
// ============================================================================

json SimulationEnv::Reset() {
    m_currentEpisode++;
    m_currentStep = 0;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "EPISODE " << m_currentEpisode << " - RESET" << std::endl;
    std::cout << "========================================" << std::endl;
    
    Simulator::Destroy();
    SetupNetwork();
    
    // Select the *FIRST* task
    m_currentTask = SelectRandomTask();
    
    // Build and return the full 5-dim state
    json state = GetState();
    json state_array = state["state"];
    
    state_array[3] = m_currentTask.inputSizeBytes / 1e6; 
    state_array[4] = m_currentTask.localProcessingTimeS;
    
    state["state"] = state_array;
    return state;
}

// ============================================================================
// UPDATE STATISTICS
// ============================================================================

void SimulationEnv::UpdateStatistics(int action, double userEnergy, double serverEnergy,
                                     double delay, double carbon, double serverCarbon) {
    if (action >= 0 && action < 3) {
        m_totalUserEnergyPerAction[action] += userEnergy;
        m_totalServerEnergyPerAction[action] += serverEnergy;
        m_totalDelayPerAction[action] += delay; // Delay is expected in ms
        m_totalCarbonPerAction[action] += (carbon + serverCarbon);
        m_actionCounts[action]++;
    }
}

// ============================================================================
// STEP FUNCTION
// ============================================================================

json SimulationEnv::Step(int action) {
    m_currentStep++;
    
    // MODIFIED: All "cheats" and "penalty blocks" have been REMOVED.
    // The agent learns from pure, natural consequences.

    // Record initial energy
    double totalEnergyBefore = 0.0;
    for (const auto& battery : m_batteries) {
        totalEnergyBefore += battery->GetRemainingEnergy();
    }
    
    double processingDelay = 0.0;
    double totalDelay = 0.0;
    double devicePowerW = m_config.deviceIdlePowerW;
    double serverEnergyJ = 0.0;
    double serverPowerW = 0.0;
    Address destination;
    
    double stepSimTime = 0.0; 
    double transmitDuration = 0.0; 

    // ACTION EXECUTION
    if (action == 0) {
        // LOCAL
        devicePowerW = m_config.deviceComputeHeavyW;
        double throttleFactor = m_thermalModels[0].GetThrottleFactor();
        processingDelay = m_currentTask.localProcessingTimeS / throttleFactor;
        stepSimTime = processingDelay;
        
    } else {
        // EDGE (1) or CLOUD (2)
        devicePowerW = m_config.deviceTxPowerWiFiW;
        
        transmitDuration = (m_currentTask.inputSizeBytes * 8.0) / 
                           (m_config.wifiDataRateMbps * 1e6);

        if (action == 1) {
    // EDGE - Use WiFi AP address (where user devices are connected)
            destination = InetSocketAddress(m_apInterface.GetAddress(0), 9000);
            processingDelay = m_edgeServer.ProcessTask(m_currentTask, m_currentTask.edgeSpeedup);
            serverPowerW = m_edgeServer.GetPerUserPower();
    
        } else { // action == 2
            // CLOUD - Use the cloud's P2P interface address
            destination = InetSocketAddress(m_edgeCloudInterface.GetAddress(1), 9000);
            processingDelay = m_cloudServer.ProcessTask(m_currentTask, m_currentTask.cloudSpeedup);
            serverPowerW = m_cloudServer.GetPerUserPower();
        }
        
        serverEnergyJ = serverPowerW * processingDelay;
        
        // Send data
        Ptr<Node> sourceNode = m_userDevices.Get(0);
        OnOffHelper onoff("ns3::UdpSocketFactory", destination);
        onoff.SetAttribute("PacketSize", UintegerValue(1400));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate(m_config.wifiDataRateMbps * 1e6)));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        
        ApplicationContainer app = onoff.Install(sourceNode);
        app.Start(Simulator::Now() + MilliSeconds(1));
        app.Stop(Simulator::Now() + Seconds(transmitDuration) + MilliSeconds(10));
        
        stepSimTime = transmitDuration + processingDelay;
    }
    
    // --- Run simulation FOR THIS TASK'S DURATION ---
    Simulator::Stop(Seconds(stepSimTime));
    Simulator::Run();
    
    // --- NOW, MEASURE WHAT HAPPENED ---
    m_flowMonitor->CheckForLostPackets();
    auto stats = m_flowMonitor->GetFlowStats();
    
    double measuredNetworkDelay = 0.0;
    double measuredPacketLoss = 0.0;
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;

    if (action == 1 || action == 2) {
        bool flowFound = false;
        
        Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowHelper.GetClassifier());

        for (auto const& [flowId, flowStats] : stats) {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);

            if (t.sourceAddress == m_staInterfaces.GetAddress(0) &&
                t.destinationAddress == InetSocketAddress::ConvertFrom(destination).GetIpv4()) 
            {
                txPackets = flowStats.txPackets;
                rxPackets = flowStats.rxPackets;
                
                if (rxPackets > 0) {
                    measuredNetworkDelay = flowStats.delaySum.GetSeconds() / rxPackets;
                }
                if (txPackets > 0) {
                    measuredPacketLoss = (double)(txPackets - rxPackets) / txPackets;
                }
                
                if (rxPackets == 0 && txPackets > 0) {
                    measuredNetworkDelay = transmitDuration; 
                }
                flowFound = true;
                break;
            }
        }
        if (!flowFound && txPackets == 0) {
            measuredNetworkDelay = transmitDuration;
        }
    }
    
    m_flowMonitor->SerializeToXmlFile("flowmon-stats.xml", true, true);

    // Total delay = upload_delay + processing_delay
    totalDelay = measuredNetworkDelay + processingDelay;
    
    // Add the return-trip WAN delay ONLY for the cloud
    if (action == 2) {
        totalDelay += (m_config.wanDelayMeanMs / 1000.0);
    }

    // Calculate energy
    double deviceEnergyJ = devicePowerW * stepSimTime;
    
    for (size_t i = 0; i < m_batteries.size(); ++i) {
        m_batteries[i]->ConsumeEnergy(deviceEnergyJ / m_batteries.size());
        m_thermalModels[i].UpdateTemperature(devicePowerW, stepSimTime);
    }
    
    double totalEnergyAfter = 0.0;
    for (const auto& battery : m_batteries) {
        totalEnergyAfter += battery->GetRemainingEnergy();
    }
    double measuredDeviceEnergy = totalEnergyBefore - totalEnergyAfter;
    
    // Carbon
    double deviceCarbonIntensity = CarbonCalculator::GetCurrentCarbonIntensity(
        m_config.carbonRegion, m_config.carbonTimeVariation);
    double deviceCarbonRate = CarbonCalculator::ConvertToPerJoule(deviceCarbonIntensity);
    
    double serverCarbonIntensity = deviceCarbonIntensity;
    if (action == 2) serverCarbonIntensity *= 0.5; // Assume cloud is greener
    double serverCarbonRate = CarbonCalculator::ConvertToPerJoule(serverCarbonIntensity);
    
    double deviceCarbonKg = measuredDeviceEnergy * deviceCarbonRate;
    double serverCarbonKg = serverEnergyJ * serverCarbonRate;
    double totalCarbonKg = deviceCarbonKg + serverCarbonKg;
    
    // Reward (PURE, no shaping)
    double energyPenalty = measuredDeviceEnergy;
    double delayPenalty = totalDelay * 1000 * m_config.rewardWeightDelay;
    double serverPenalty = serverEnergyJ * m_config.rewardWeightServerEnergy;
    double carbonPenalty = totalCarbonKg * 1e6 * m_config.rewardWeightCarbon;

    // --- MODIFIED: Add a "Survival Bonus" ONLY for Action 0 ---
    // This encourages the agent to explore "Local" without fear.
    double stepBonus = 0.0; 
    if (action == 0) {
        stepBonus = 70.0; 
    }

    double reward = stepBonus - (energyPenalty + delayPenalty + serverPenalty + carbonPenalty);
    
    // Update statistics
    UpdateStatistics(action, measuredDeviceEnergy, serverEnergyJ, 
                     totalDelay * 1000, deviceCarbonKg, serverCarbonKg);
    
    // Check if done
    bool done = (m_currentStep >= m_config.simulationStepsPerEpisode);
    for (const auto& battery : m_batteries) {
        if (battery->IsDepleted()) {
            done = true;
            break;
        }
    }
    
    // Print progress every 20 steps
    if (m_currentStep % 20 == 0 || done) {
        std::cout << "Step " << m_currentStep << "/" << m_config.simulationStepsPerEpisode
                  << " | Action: " << action 
                  << " | Task: " << m_currentTask.name
                  << " | Reward: " << std::fixed << std::setprecision(2) << reward
                  << " | Delay(ms): " << (totalDelay * 1000)
                  << " | PktLoss: " << measuredPacketLoss
                  << " | Battery: " << (m_batteries[0]->GetStateOfCharge() * 100) << "%"
                  // --- COMPILE ERROR FIX ---
                  // Use '.' instead of '->' because m_thermalModels holds objects, not pointers
                  << " | Temp: " << m_thermalModels[0].GetTemperature() << "°C" << std::endl;
    }
    
    // --- Build the final response ---
    
    // 1. Get the base 3-dim state
    json nextStateBase = GetState();
    
    // 2. Select the *NEXT* task for the agent to see
    m_currentTask = SelectRandomTask(); 
    
    // 3. Build the full 5-dim state to return
    json nextStateArray = nextStateBase["state"];
    nextStateArray[3] = m_currentTask.inputSizeBytes / 1e6; // Normalized
    nextStateArray[4] = m_currentTask.localProcessingTimeS;

    json result;
    result["next_state"] = nextStateArray;
    result["reward"] = reward;
    result["done"] = done;
    
    return result;
}

// ============================================================================
// PRINT PROGRESS REPORT
// ============================================================================

void SimulationEnv::PrintProgressReport() {
    std::cout << "\n--- Episode " << m_currentEpisode << " Summary ---" << std::endl;
    std::cout << "Actions taken: Local=" << m_actionCounts[0] 
              << ", Edge=" << m_actionCounts[1] 
              << ", Cloud=" << m_actionCounts[2] << std::endl;
}

// ============================================================================
// PRINT COMPARISON REPORT
// ============================================================================

void SimulationEnv::PrintComparisonReport() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL STATISTICS ACROSS ALL EPISODES" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<std::string> actionNames = {"Local", "Edge", "Cloud"};
    
    for (int i = 0; i < 3; ++i) {
        if (m_actionCounts[i] == 0) continue;
        
        std::cout << "\n--- " << actionNames[i] << " Processing ---" << std::endl;
        std::cout << "Count: " << m_actionCounts[i] << std::endl;
        std::cout << "Avg User Energy: " << (m_totalUserEnergyPerAction[i] / m_actionCounts[i]) << " J" << std::endl;
        std::cout << "Avg Server Energy: " << (m_totalServerEnergyPerAction[i] / m_actionCounts[i]) << " J" << std::endl;
        std::cout << "Avg Delay: " << (m_totalDelayPerAction[i] / m_actionCounts[i]) << " ms" << std::endl;
        std::cout << "Avg Carbon: " << (m_totalCarbonPerAction[i] / m_actionCounts[i] * 1e6) << " mg CO2" << std::endl;
    }
}

// ============================================================================
// MAIN FUNCTION - ZMQ SERVER
// ============================================================================

int main(int argc, char* argv[]) {
    // This is a good place to turn off logging if it gets too noisy
    //LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
    //LogComponentEnable("PacketSink", LOG_LEVEL_INFO);

    std::cout << "========================================" << std::endl;
    std::cout << "NS-3 DRL Simulation Server" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create configuration
    SimConfig config;
    config.numUserDevices = 5;
    config.simulationStepsPerEpisode = 100;
    config.carbonRegion = "US-California";
    config.packetLossRate = 0.01;
    
    // --- THIS IS THE NEW, BALANCED PHYSICS ---
    
    // 1. LOCAL COMPUTE is cheap
    config.deviceComputeHeavyW = 0.3; 

    // 2. CLOUD DELAY is significant but beatable
    config.wanDelayMeanMs = 300.0; // 300ms WAN delay

    // 3. REWARDS: Delay penalty is reduced
    config.rewardWeightDelay = 0.05; 
    config.rewardWeightServerEnergy = 0.1;
    config.rewardWeightCarbon = 0.1;

    // 4. *** LOGICAL TASK PHYSICS ***
    // This physics creates the trade-offs you want.
    config.taskTypes = {
        // name, probability, size, time(s), energy(J), edge_speed, cloud_speed, isExtreme
        
        // GOAL: LOCAL IS BEST (5ms)
        {"light", 0.4, 100000, 0.005, 1.0, 2.0, 5.0, false}, 
        
        // GOAL: EDGE IS BEST
        {"medium", 0.3, 200000, 2.0, 2.5, 6.0, 12.0, false},
        
        // GOAL: CLOUD IS BEST
        {"heavy", 0.3, 500000, 20.0, 5.0, 10.0, 100.0, true} // 3 tasks, probs sum to 1.0
    };
    
    // --- END OF NEW PHYSICS ---
    
    // Initialize simulation environment
    SimulationEnv env(config);
    
    // Setup ZMQ Server
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind("tcp://*:5555");
    
    std::cout << "ZMQ Server listening on port 5555..." << std::endl;
    std::cout << "Waiting for Python DRL client connection..." << std::endl;
    
    bool running = true;
    
    while (running) {
        zmq::message_t request;
        
        // Receive request from Python
        auto result = socket.recv(request, zmq::recv_flags::none);
        if (!result) continue;
        
        std::string request_str(static_cast<char*>(request.data()), request.size());
        json request_json = json::parse(request_str);
        
        std::string command = request_json["command"];
        json response;
        
        if (command == "reset") {
            response = env.Reset();
            
        } else if (command == "step") {
            int action = request_json["action"];
            response = env.Step(action);
            
        } else if (command == "shutdown") {
            std::cout << "\nReceived shutdown command." << std::endl;
            env.PrintComparisonReport();
            response["status"] = "shutdown";
            running = false;
            
        } else {
            response["error"] = "Unknown command";
        }
        
        // Send response back to Python
        std::string response_str = response.dump();
        zmq::message_t reply(response_str.size());
        memcpy(reply.data(), response_str.data(), response_str.size());
        socket.send(reply, zmq::send_flags::none);
    }
    
    std::cout << "Simulation server shutting down..." << std::endl;
    Simulator::Destroy();
    
    return 0;
}