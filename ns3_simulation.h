#ifndef NS3_SIMULATION_ENV_H // Changed guard name slightly for clarity
#define NS3_SIMULATION_ENV_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h" // Include headers for modules used
#include "ns3/applications-module.h"
#include "ns3/energy-module.h"
#include "ns3/csma-module.h"

#include <nlohmann/json.hpp>
#include <vector>
#include <random>

// Use namespaces for convenience
using namespace ns3;
using json = nlohmann::json; // Alias for nlohmann::json

class SimulationEnv {
public:
    SimulationEnv(); // Constructor

    // Public methods called by main ZMQ loop
    json Reset();
    json Step(int action);
    json GetState(); // Added GetState

private:
    // Internal method to set up the network
    void SetupNetwork();

    // Random number generator for mocked parts
    std::mt19937 m_rng;

    // NS-3 objects containers
    NodeContainer m_user_devices;
    NodeContainer m_edge_node;
    NodeContainer m_cloud_node;
    // Store Ptrs to energy sources and models
    std::vector<Ptr<ns3::energy::BasicEnergySource>> m_energy_sources;
    std::vector<Ptr<ns3::energy::SimpleDeviceEnergyModel>> m_device_models;

    // Simulation state variables
    int m_current_step;
    double m_edge_load;    // Mocked
    double m_cloud_load;   // Mocked
    double m_last_delay;   // Mocked
};

#endif // NS3_SIMULATION_ENV_H