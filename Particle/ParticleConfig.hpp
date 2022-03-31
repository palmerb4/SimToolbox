/**
 * @file ParticleConfig.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief read configuration parameters from a yaml file
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef PARTICLECONFIG_HPP_
#define PARTICLECONFIG_HPP_

#include "Boundary/Boundary.hpp"
#include "Particle/ParticleShapes.hpp"
#include "Util/GeoCommon.h"

#include <iostream>
#include <memory>

/**
 * @brief read configuration parameters from a yaml file
 *
 */
class ParticleConfig {
  public:
    unsigned int rngSeed; ///< random number seed
    int logLevel;         ///< follows SPDLOG level enum, see Util/Logger.hpp for details
    int timerLevel = 0;   ///< how detailed the timer should be

    // domain setting
    double simBoxHigh[3];   ///< simulation box size
    double simBoxLow[3];    ///< simulation box size
    bool simBoxPBC[3];      ///< flag of true/false of periodic in that direction
    bool monolayer = false; ///< flag for simulating monolayer on x-y plane

    double initBoxHigh[3];      ///< initialize particles within this box
    double initBoxLow[3];       ///< initialize particles within this box
    double initOrient[3];       ///< initial orientation for each particle. >1 <-1 means random
    bool initCircularX = false; ///< set the initial cross-section as a circle in the yz-plane
    int initPreSteps = 100;     ///< number of initial pre steps to resolve potential collisions

    // physical constant
    double viscosity; ///< unit pN/(um^2 s), water ~ 0.0009
    double KBT;       ///< pN.um, 0.00411 at 300K
    double linkKappa; ///< pN/um stiffness of particle links
    double linkGap;   ///< um length of gap between particle links

    // particle settings
    bool particleFixed = false; ///< particles do not move
    int particleNumber;         ///< initial number of particles
    double particleLength;      ///< particle length (mean if sigma>0)
    double particleLengthSigma; ///< particle length lognormal distribution sigma
    double particleDiameter;    ///< particle diameter

    // external forcing
    double externalForce[3];   ///< simulation box size
    double externalTorque[3];   ///< simulation box size

    // collision radius and diameter
    double particleDiameterColRatio; ///< collision diameter = ratio * real diameter
    double particleLengthColRatio;   ///< collision length = ratio * real length
    double particleColBuf;           ///< threshold for recording possible collision

    // time stepping
    double dt;        ///< timestep size
    double timeTotal; ///< total simulation time
    double timeSnap;  ///< snapshot time. save one group of data for each snapshot

    // constraint solver
    double conResTol;    ///< constraint solver residual
    int conMaxIte;       ///< constraint solver maximum iteration
    int conSolverChoice; ///< choose a iterative solver. 0 for BBPGD, 1 for APGD, etc

    // hydro
    std::string prescribedSurfaceDensityFile;
    
    std::vector<std::shared_ptr<Boundary>> boundaryPtr;
    std::shared_ptr<const ParticleShapes> particleShapesPtr;

    ParticleConfig() = default;
    ParticleConfig(std::string filename);
    ~ParticleConfig() = default;

    void dump() const;
};

#endif