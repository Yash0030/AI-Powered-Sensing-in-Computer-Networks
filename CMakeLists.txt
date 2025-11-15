cmake_minimum_required(VERSION 3.16)
project(ns3_drl_sim LANGUAGES CXX)
include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)

# -----------------------
# Find ZeroMQ
# -----------------------
pkg_check_modules(ZMQ REQUIRED libzmq)

# -----------------------
# Find nlohmann_json
# -----------------------
find_package(nlohmann_json 3.2.0 REQUIRED)

# -----------------------
# Find ns-3 (Requires CMAKE_PREFIX_PATH pointing to ns-3 install dir)
# Example: cmake .. -DCMAKE_PREFIX_PATH=~/ns-3-dev/install
# -----------------------
find_package(ns3 REQUIRED COMPONENTS
    core
    network
    internet
    point-to-point
    applications
    energy
    csma
    wifi              # ✅ ADDED - Required for realistic WiFi
    mobility          # ✅ ADDED - Required for device movement
    propagation       # ✅ ADDED - Required for wireless propagation
    spectrum          # ✅ ADDED - Required for WiFi spectrum
    flow-monitor      # ✅ ADDED - For network statistics
)

# -----------------------
# Add executable
# -----------------------
# ⚠️ UPDATE THIS: Change filename to match your actual C++ file
add_executable(ns3_drl_sim ns3_simulation.cpp)  # Changed from ns3_simulation.cpp

# -----------------------
# Link libraries (Using modern CMake targets)
# -----------------------
target_link_libraries(ns3_drl_sim
    PRIVATE
    ${ZMQ_LIBRARIES}
    nlohmann_json::nlohmann_json
    # NS-3 module targets
    ns3::core
    ns3::network
    ns3::internet
    ns3::point-to-point
    ns3::applications
    ns3::energy
    ns3::csma
    ns3::wifi              # ✅ ADDED
    ns3::mobility          # ✅ ADDED
    ns3::propagation       # ✅ ADDED
    ns3::spectrum          # ✅ ADDED
    ns3::flow-monitor      # ✅ ADDED
)

# -----------------------
# Compiler settings
# -----------------------
target_compile_features(ns3_drl_sim PRIVATE cxx_std_23)  # C++23 for ns-3-dev
target_compile_options(ns3_drl_sim PRIVATE 
    -Wall 
    -Wextra 
    -Wno-unused-parameter
    -O2  # ✅ ADDED - Optimization level 2
)

# -----------------------
# Include ZMQ directories (from pkg-config)
# -----------------------
target_include_directories(ns3_drl_sim PRIVATE ${ZMQ_INCLUDE_DIRS})

# -----------------------
# Installation rule
# -----------------------
install(TARGETS ns3_drl_sim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# -----------------------
# Optional: Print configuration summary
# -----------------------
message(STATUS "========================================")
message(STATUS "NS-3 DRL Simulation Configuration")
message(STATUS "========================================")
message(STATUS "C++ Standard: C++23")
message(STATUS "ZMQ Found: ${ZMQ_FOUND}")
message(STATUS "nlohmann_json Found: ${nlohmann_json_FOUND}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")