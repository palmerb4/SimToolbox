#include "HydroParticleMobility.hpp"

// TODO: 
//  1. (done) apply needs updated to match the wrapper syntax.

template <class Container>
HydroParticleMobility<Container>::HydroParticleMobility(const Container *const containerPtr, const int nParticleLocal,
                                                    const ParticleConfig &runConfig)
    : containerPtr(containerPtr), nParticleLocal(nParticleLocal), runConfig(runConfig) {
    commRcp = getMPIWORLDTCOMM();
    mobMapRcp = getTMAPFromLocalSize(6 * nParticleLocal, commRcp);
    AOpRcp = Teuchos::rcp(new HydroParticleOperator<Container>(containerPtr, nParticleLocal, runConfig));
    auto xMapRcp = AOpRcp->getDomainMap();
    auto bMapRcp = AOpRcp->getRangeMap();

    xRcp = Teuchos::rcp(new TV(xMapRcp, true));
    bRcp = Teuchos::rcp(new TV(bMapRcp, true));
    xVec.reserve(6 * nParticleLocal);
    bVec.reserve(6 * nParticleLocal);

    // setup the problem
    problemRcp = Teuchos::rcp(new Belos::LinearProblem<TOP::scalar_type, TMV, TOP>(AOpRcp, xRcp, bRcp));

    Belos::SolverFactory<TOP::scalar_type, TMV, TOP> factory;

    // choose solver in constructor
    setSolverParameters();
    solverRcp = factory.create(solverParamsRcp->name(), solverParamsRcp);
    solverRcp->setProblem(problemRcp);
    if (commRcp->getRank() == 0) {
        std::cout << "Iterative Solver: " << solverParamsRcp->name() << std::endl;
    }
}

template <class Container>
void HydroParticleMobility<Container>::setSolverParameters() {
    std::string fileName = "mobilitySolver.yaml";
    if (IOHelper::fileExist(fileName))
        solverParamsRcp = Teuchos::getParametersFromYamlFile(fileName);
    else {
        // use default parameters
        solverParamsRcp = Teuchos::parameterList();
        solverParamsRcp->setName("GMRES");
        solverParamsRcp->set("Num Blocks", 30);
        solverParamsRcp->set("Maximum Restarts", 20);
        solverParamsRcp->set("Maximum Iterations", 1000);
        solverParamsRcp->set("Convergence Tolerance", 1e-8);
        solverParamsRcp->set("Timer Label", "HydroParticleMobility");
        solverParamsRcp->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::TimingDetails + Belos::FinalSummary);
        solverParamsRcp->set("Output Frequency", -1); // 1 for every iteration
        solverParamsRcp->set("Show Maximum Residual Norm Only", false);
        solverParamsRcp->set("Implicit Residual Scaling", "Norm of RHS");
        solverParamsRcp->set("Explicit Residual Scaling", "Norm of RHS");
    }
}

template <class Container>
void HydroParticleMobility<Container>::testOperator() {

    // calc vel for some x
    Teuchos::RCP<TV> vwVecRcp = Teuchos::rcp(new TV(mobMapRcp, true));
    xRcp->putScalar(0.0);
    AOpRcp->calcVelOmega(*xRcp, *vwVecRcp, option);
    dumpTV(vwVecRcp, "VelOmegaZero");

    // dump motion
    calcMotion(*vwVecRcp);
    dumpTV(bRcp, "sol_bVec");
    dumpTV(xRcp, "sol_x");
    dumpTV(vwVecRcp, "sol_VelOmega");
}

template <class Container>
void HydroParticleMobility<Container>::cacheResults(Container &particleContainer) const {
    AOpRcp->cacheResults(particleContainer);
}

template <class Container>
void HydroParticleMobility<Container>::calcMotion() {
    // for calcMotion we use both the user defined total force/torque
    // as well as the user defined surface densities. These quantities are ADDITIVE,
    // as the user defined surface density need not be force/torque free.

    // get user input values (use b as temporary storage for the userInputSurfaceDensityCoeff)
    AOpRcp->getUserInputSurfaceDensityCoeffWrapper(*bRcp);

    // calc contribution to rhs from user input
    AOpRcp->measurebWrapper(*bRcp, *bRcp);

    // populate initial guess using data stored in particleContainer
    AOpRcp->getSurfaceDensityCoeffFromParticleContainerWrapper(*xRcp);

    // setup initial guess and right hand side
    bool set = problemRcp->setProblem(xRcp, bRcp);
    TEUCHOS_TEST_FOR_EXCEPTION(!set, std::runtime_error, "*** Belos::LinearProblem failed to set up correctly! ***");

    // solve Ax=b
    Belos::ReturnType result = solverRcp->solve();
    int numIters = solverRcp->getNumIters();
    if (commRcp->getRank() == 0) {
        std::cout << "RECORD: Num of Iterations in Mobility Matrix: " << numIters << std::endl;
    }

    // post-process results. Results are stored in the operator.
    AOpRcp->calcParticleVelOmega();
}

template <class Container>
void HydroParticleMobility<Container>::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode, scalar_type alpha,
                                           scalar_type beta) const {
    // X: ForceTorque
    // Y: VelOmega
    // compute Y=alpha*Ax+beta*Y;
    assert(X.getMap()->isSameAs(*(Y.getMap())));
    assert(X.getMap()->isSameAs(*mobMapRcp));
    assert(X.getNumVectors() == Y.getNumVectors());

    Teuchos::RCP<TV> YColOld = Teuchos::rcp(new TV(Y.getMap(), true));
    HydroOption option;
    option.withSwim = false;
    option.withForceTorqueExternal = false;

    const int nCol = X.getNumVectors();
    for (int c = 0; c < nCol; c++) {
        const auto &XCol = X.getVector(c);
        auto YCol = Y.getVectorNonConst(c);
        YColOld->update(beta, *YCol, 0.0); // Yold = beta*Ycol

        // Setup the RHB
        // get the surface density generated by the input force/torque
        // (use b as temporary storage)
        AOpRcp->measureBxWrapper(*XCol, *bRcp);

        // calc contribution to rhs from input force/torque
        AOpRcp->measurebWrapper(*bRcp, *bRcp);

        // compute YCol=A*XCol
        bool set = problemRcp->setProblem(); // iterative solve
        TEUCHOS_TEST_FOR_EXCEPTION(!set, std::runtime_error,
                                   "*** Belos::LinearProblem failed to set up correctly! ***");
        solverRcp->reset(Belos::Problem);
        solverRcp->setProblem(problemRcp);

        Belos::ReturnType result = solverRcp->solve();
        int numIters = solverRcp->getNumIters();
        if (commRcp->getRank() == 0) {
            std::cout << "RECORD: Num of Iterations in Mobility Matrix: " << numIters << std::endl;
        }
        AOpRcp->calcVelOmega(*xRcp, *YCol);

        YCol->update(1.0, *YColOld, alpha); // Ycol = alpha*AXcol+beta*Ycol
    }
}
