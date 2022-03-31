/**
 * @file ParticleSystem_main.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief main driver of ParticleSystem class
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <mpi.h>

#include "ParticleSystem.hpp"
#include "Util/Logger.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    PS::Initialize(argc, argv);
    Logger::setup_mpi_spdlog();
    {
        // create a system and distribute it to all ranks
        // MPI is initialized inside PS::Initialize()
        std::string runConfig = "RunConfig.yaml";
        std::string posFile = "ParticleInitial.dat";
        std::string restartFile = "TimeStepInfo.txt";
        constexpr int spectralDegree = 6;
        ParticleSystem<spectralDegree> system;

        if (IOHelper::fileExist(restartFile)) {
            system.reinitialize(runConfig, restartFile, argc, argv, true);
        } else {
            system.initialize(runConfig, posFile, argc, argv);
        }

        // main time loop
        while (system.getStepCount() * system.runConfig.dt < system.runConfig.timeTotal) {
            system.prepareStep();
            system.runStep();
            system.calcOrderParameter();
            system.calcConStress();
            system.printTimingSummary();
        }
    }
    // mpi finalize
    // let the root rank wait for other
    PS::Finalize();
    MPI_Finalize();
    return 0;
}