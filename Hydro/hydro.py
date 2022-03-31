import os
import time
import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.spatial.transform import Rotation
from scipy.special import eval_legendre
from scipy.io import mmwrite
from collision import Collision


def calcAnalyticalFlowMicroHydroTranslation(theta, phi, a=1.0, eta=1.0, pinf=0.0, U = np.array([1.0, 0.0, 0.0])):
    def kdelta(i, j):
        return 1 if i == j else 0

    # sphere basis
    rHat = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    thetaHat = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    phiHat = np.array([-np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), 0])

    # analytical solution
    xyz = rHat
    r = np.linalg.norm(xyz)
    uVec = np.zeros(3)
    fVec = np.zeros(3)
    totalForce = np.zeros(3)
    totalTorque = np.zeros(3)
    totalForce = 6 * np.pi * eta * a * U
    fVec = pinf * rHat + 3 * eta * U / 2 / a
    for i in range(3):
        for j in range(3):
            uVec[i] += 3 * a / 4 / r * (kdelta(i, j) + xyz[i] * xyz[j] / r**2) * U[j] \
                    - a**3 / 4 / r**3 * (-kdelta(i, j) + 3 * xyz[i] * xyz[j] / r**2) * U[j]

    return uVec, fVec, totalForce, totalTorque

def calcAnalyticalFlowMicroHydroRotation(theta, phi, a=1.0, eta=1.0, pinf=0.0, Sigma = np.array([1.0, 0.0, 0.0])):
    def e(i, j, k):
        ijk = np.array([i, j, k])
        return np.sum(np.sign(np.roll(ijk, -1) - ijk))

    # sphere basis
    rHat = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    thetaHat = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    phiHat = np.array([-np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), 0])

    # analytical solution
    xyz = rHat
    r = np.linalg.norm(xyz)
    uVec = np.zeros(3)
    totalForce = np.zeros(3)
    totalTorque = np.zeros(3)
    totalTorque = 8 * np.pi * eta * a**3 * Sigma
    for i in range(3):
        for j in range(3):
            for k in range(3):
                uVec[i] += a**3 * e(i,j,k) * Sigma[j] * xyz[k] / r**3

    return uVec, None, totalForce, totalTorque


def calcAnalyticalFlowSphere(xyz, mu=1.0, a=1.0):
    def e(i, j, k):
        ijk = np.array([i, j, k])
        return np.sum(np.sign(np.roll(ijk, -1) - ijk))

    def kdelta(i, j):
        return 1 if i == j else 0

    def s(j, k):
        return kdelta(2, j) * kdelta(0, k) + kdelta(0, j) * kdelta(2, k)


    uVec = np.zeros(3)
    fVec = np.zeros(3)
    for i in range(3):
        r = np.linalg.norm(xyz)
        fVec[i] = mu / a * (4 * kdelta(0, i) * xyz[2] + kdelta(2, i) * xyz[0])
        for k in range(3):
            uVec[i] += (1 / 2 - a**3 / (2 * r**3)) * e(i, 1, k) * xyz[k]
            for j in range(3):
                uVec[i] += (5 * a**5 / (4 * r**7) - 5 * a**3 / (4 * r**5)) * xyz[i] * xyz[j] * xyz[k] * s(j, k)
        for j in range(3):
            uVec[i] += (1 / 2 - a**5 / (2 * r**5)) * s(i, j) * xyz[j]

    return (uVec, fVec)


# def calcAnalyticalFlowSphere(U3=1, U2=0, mu=0.01, a=1.0, x=[], y=[], z=[]):
#     # Hydromechanics of low-Reynolds-number flow. Part 2. Singularity method for Stokes flows 
#     # Chwang, A. T., & Wu, T. Y.-T. (1975)
#     # We align with the z-axis, whereas Chwang aligns with the x-axis
#     from sympy.tensor.array import derive_by_array
#     import matplotlib.pyplot as plt
#     import sympy as sym
#     import numpy as np

#     # create symbolic variables
#     x_ = sym.Symbol('x')
#     y_ = sym.Symbol('y')
#     z_ = sym.Symbol('z')
#     U3_ = sym.Symbol('U3')
#     U2_ = sym.Symbol('U2')
#     mu_ = sym.Symbol('mu')
#     a_ = sym.Symbol('a')

#     # shape properties
#     r_ = sym.sqrt(y_**2 + x_**2)
#     R_ = sym.sqrt(z_**2 + r_**2)

#     # solve for the flow at the point x,y,z
#     ex_ = sym.Array([1,0,0])
#     ey_ = sym.Array([0,1,0])
#     ez_ = sym.Array([0,0,1])
#     xVec_ = sym.Array([x_,y_,z_])
#     UVec_ = U3_ * ez_ + U2_ * ey_
#     u_ = UVec_ - 3 * a_ / 4 * (UVec_ / R_ + (U3_ * z_ + U2_ * y_) * xVec_ / R_**3)

#     u1 = np.zeros(len(x)).astype(complex)
#     u2 = np.zeros(len(x)).astype(complex)
#     u3 = np.zeros(len(x)).astype(complex)

#     # Notive that this is where we make the switch from x-aligned to z-aligned
#     for i in range(len(x)):
#         u3[i] = u_[0].subs({x_:x[i], y_:y[i], z_:z[i], U3_:U3, U2_:U2, mu_:mu, a_:a})
#         u2[i] = u_[1].subs({x_:x[i], y_:y[i], z_:z[i], U3_:U3, U2_:U2, mu_:mu, a_:a})
#         u1[i] = u_[2].subs({x_:x[i], y_:y[i], z_:z[i], U3_:U3, U2_:U2, mu_:mu, a_:a})

#     # solve for the total force and torque experienced by the particle
#     coeff = 6 * np.pi * mu * a
#     externalForceTorque = np.zeros(6)
#     externalForceTorque[0] = 0
#     externalForceTorque[1] = coeff * U2
#     externalForceTorque[2] = coeff * U3
    
#     return u1, u2, u3, externalForceTorque


# def calcAnalyticalFlowSphere(U=1, mu=0.01, radius=1.0, xyz=[]):
#     R = np.linalg.norm(xyz)
#     theta = np.arctan2(np.sqrt(xyz[0]**2 + xyz[1]**2), xyz[2])
#     phi = np.arctan2(xyz[1], xyz[0])
#     ur = -U * np.cos(theta) * (1 - 3 * radius / (2 * R) + radius**3 / (2 * R**3))
#     utheta = U * np.sin(theta) * (1 - 3 * radius / (4 * R) - radius**3 / (4 * R**3))
#     rHat = xyz / R
#     rHat = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
#     thetaHat = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])

#     uVec = ur * rHat + utheta * thetaHat 
#     externalForceTorque = np.array([0.0, 0.0, 6 * np.pi * mu * U * radius, 0.0, 0.0, 0.0])
#     return uVec[0], uVec[1], uVec[2], externalForceTorque


# def calcAnalyticalFlowProlateSpheroid(U3=1, U2=0, mu=0.01, a=1.0, b=0.5, x=[], y=[], z=[]):
#     # Hydromechanics of low-Reynolds-number flow. Part 2. Singularity method for Stokes flows 
#     # Chwang, A. T., & Wu, T. Y.-T. (1975)
#     from sympy.tensor.array import derive_by_array
#     import matplotlib.pyplot as plt
#     import sympy as sym
#     import numpy as np

#     # create symbolic variables
#     x_ = sym.Symbol('x')
#     y_ = sym.Symbol('y')
#     z_ = sym.Symbol('z')
#     U3_ = sym.Symbol('U3')
#     U2_ = sym.Symbol('U2')
#     mu_ = sym.Symbol('mu')
#     a_ = sym.Symbol('a')
#     b_ = sym.Symbol('b')

#     # shape properties
#     e_ = sym.sqrt(a_**2 - b_**2) / a_
#     c_ = e_ * a_
#     r_ = sym.sqrt(y_**2 + x_**2)
#     R1_ = sym.sqrt((z_ + c_)**2 + r_**2)
#     R2_ = sym.sqrt((z_ - c_)**2 + r_**2)

#     # Solve for the harmonics (what type of harmonics... this I do not know)
#     B10_ = sym.log((R2_ - (z_ - c_))/(R1_ - (z_ + c_)))
#     B11_ = R2_ - R1_ + z_ * B10_
#     B30_ = 1 / r_**2 * ((z_ + c_) / R1_ - (z_ - c_) / R2_)
#     B31_ = 1 / R1_ - 1 / R2_ + z_ * B30_

#     # solve for the flow at the point x,y,z
#     ex_ = sym.Array([1,0,0])
#     ey_ = sym.Array([0,1,0])
#     ez_ = sym.Array([0,0,1])
#     er_ = (y_ * ey_ + x_ * ex_) / r_
#     Le_ = sym.log((1 + e_) / (1 - e_))
#     alpha1_ = U3_ * e_**2 * 1 / (-2 * e_ + (1 + e_**2) * Le_) 
#     alpha2_ = 2 * U2_ * e_**2 * 1 / (2 * e_ + (3 * e_**2 - 1) * Le_) 
#     beta1_ = (1 - e_**2) / (2 * e_**2) * alpha1_
#     beta2_ = (1 - e_**2) / (2 * e_**2) * alpha2_
#     CF3_ = 8 * c_ * alpha1_ /  (3 * a_ * U3_)
#     CF2_ = 8 * c_ * alpha2_ /  (3 * a_ * U2_)

#     u_ = U3_ * ez_ + U2_ * ey_ - (2 * alpha1_ * ez_ + alpha2_ * ey_) * B10_ \
#         - (alpha1_ * r_ * er_ + alpha2_ * y_ * ez_) * (1 / R2_ - 1 / R1_) \
#         + (alpha1_ * r_ * ez_ - alpha2_ * y_ * er_) * r_ * B30_ \
#         + derive_by_array(-2 * beta1_ * B11_ + beta2_ * y_ * ((z_ - c_) / r_**2 * R1_ - (z_ + c_) / r_**2 * R2_ + B10_),
#                     (x_,y_,z_))
#     # u_ = U3_ * ez_ + U2_ * ey_ - (2 * alpha1_ * ez_ + alpha2_ * ey_) * B10_ \
#     #     - (alpha1_ * r_ * er_ + alpha2_ * y_ * ez_) * (1 / R2_ - 1 / R1_) \
#     #     + (alpha1_ * r_ * ez_ - alpha2_ * y_ * er_) * r_ * B30_


#     u1 = np.zeros(len(x)).astype(complex)
#     u2 = np.zeros(len(x)).astype(complex)
#     u3 = np.zeros(len(x)).astype(complex)

#     # Notive that this is where we make the switch from x-aligned to z-aligned
#     for i in range(len(x)):
#         u1[i] = u_[0].subs({x_:x[i], y_:y[i], z_:z[i], U3_:U3, U2_:U2, mu_:mu, a_:a, b_:b})
#         u2[i] = u_[1].subs({x_:x[i], y_:y[i], z_:z[i], U3_:U3, U2_:U2, mu_:mu, a_:a, b_:b})
#         u3[i] = u_[2].subs({x_:x[i], y_:y[i], z_:z[i], U3_:U3, U2_:U2, mu_:mu, a_:a, b_:b})

#     # solve for the total force and torque experienced by the particle
#     CF3 = CF3_.subs({U3_:U3, U2_:U2, mu_:mu, a_:a, b_:b})
#     CF2 = CF2_.subs({U3_:U3, U2_:U2, mu_:mu, a_:a, b_:b})
#     coeff = 6 * np.pi * mu * a
#     externalForceTorque = np.zeros(6)
#     externalForceTorque[0] = 0
#     externalForceTorque[1] = coeff * U2 * CF2
#     externalForceTorque[2] = coeff * U3 * CF3
    
#     return u1, u2, u3, externalForceTorque


class Hydro:
    """class for evaluating the hydrodynamic interactions between particles"""

    def __init__(self, name_, runConfig_, particleContainer_):
        self.name = name_
        self.runConfig = runConfig_
        self.particleContainer = particleContainer_
        self.numPtcLocal = len(self.particleContainer)

        # indices, maps, etc, precomputed in constructor
        # Q: number of grid points per particle
        # C: number of spectral coefficients per particle
        self.particlePts = np.zeros(self.numPtcLocal, dtype=int)   # < 1 dof per particle, stores Q
        self.particleCoeffs = np.zeros(self.numPtcLocal, dtype=int)   # < 1 dof per particle, stores C
        self.particlePtsIndex = np.zeros(self.numPtcLocal + 1, dtype=int) # < beginning of grid points per particle in particlePtsMap
        self.particleCoeffIndex = np.zeros(self.numPtcLocal + 1, dtype=int) # < beginning of spectral coefficients per particle in particleCoeffMap
        for pidx in range(self.numPtcLocal):
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)
            self.particlePts[pidx] = layer.getNumGridPts()
            self.particleCoeffs[pidx] = layer.getNumSpectralCoeff()
            self.particlePtsIndex[pidx + 1] = self.particlePtsIndex[pidx] + self.particlePts[pidx]
            self.particleCoeffIndex[pidx + 1] = self.particleCoeffIndex[pidx] + self.particleCoeffs[pidx]
            
        # self.particleMapRcp = np.zeros(self.numPtcLocal)  # < 1 dof per particle
        # self.particlePtsMapRcp = np.zeros(self.particlePtsIndex[-1]) # < Q dof per particle
        # self.particleCoeffMapRcp = np.zeros(self.particleCoeffIndex[-1]) # < C dof per particle
        # self.pointValuesMapRcp = np.zeros(3 * self.particlePtsIndex[-1]) # < 3Q dof per particle
        # self.pointCoeffMapRcp = np.zeros(3 * self.particleCoeffIndex[-1]) # < 3C dof per particle

        # initialize storage
        self.collisionForceTorque = np.zeros(6 * self.numPtcLocal) #< force/torque from collision 6 dof per particle
        self.externalForceTorque = np.zeros(6 * self.numPtcLocal) #< external force/torque specified in RunConfig, 6 dof per particle

        # self.externalForceTorque[0::3] = self.runConfig.externalForce[0]
        # self.externalForceTorque[1::3] = self.runConfig.externalForce[1]
        # self.externalForceTorque[2::3] = self.runConfig.externalForce[2]
        # self.externalForceTorque[3::3] = self.runConfig.externalTorque[0]
        # self.externalForceTorque[4::3] = self.runConfig.externalTorque[1]
        # self.externalForceTorque[5::3] = self.runConfig.externalTorque[2]

        print("externalForceTorque are hardcoded")
        a = 1.0
        direction = np.array([0.0, -1.0, 0.0])
        self.externalForceTorque[0:3] = 6 * np.pi * a * direction

        # for testing purposes
        # print("DELETE ME")
        # _, _, _, self.externalForceTorque = calcAnalyticalFlowProlateSpheroid(U3=1, U2=0, mu=self.runConfig.viscosity, 
        #     a=self.runConfig.particleAxisLen3, b=self.runConfig.particleAxisLen2) # [0.0, 0.0, 11.346876506622714]
        # _, _, _, self.externalForceTorque = calcAnalyticalFlowSphere(U3=1, U2=0, mu=self.runConfig.viscosity, 
        #     a=self.runConfig.particleRadius) # [0.0, 0.0, 18.84955592153876]


        # precompute the J matrix to save some time when computing A
        self.JMatrix = self.calcJMatrix()

        # setup the linear problem Ax = b where x = unknownSurfaceDensityCoeff
        N = 3 * self.particleCoeffIndex[-1]
        M = 3 * self.particlePtsIndex[-1]
        self.knownSurfaceDensityCoeff = np.zeros(N, dtype=np.complex128)
        self.knownSurfaceDensityGridVals = np.zeros(M)
        self.unknownSurfaceDensityCoeff = np.zeros(N, dtype=np.complex128)
        self.unknownSurfaceDensityGridVals = np.zeros(M)
        self.measureAop = LinearOperator(shape=(N, N), matvec=self.measureAx)
        self.AxCoeff = np.zeros(N, dtype=np.complex128)

    def calcFTc(self):
        collision = Collision(self.runConfig, self.runConfig, self.particleContainer)
        collision.calcFTc()
        return collision.collisionForceTorque         

    def run(self):
        """calculate the centroid velocity of each particle induced by the hydrodynamic interactions and external force/torque"""

        # solve collision force and torque 
        # TODO update collision solver 
        # self.collisionForceTorque = self.calcFTc()

        # solve for b
        tic = time.perf_counter()
        userInputSurfaceDensityCoeff = self.getUserInputSurfaceDensity()
        rho = self.measureLx(userInputSurfaceDensityCoeff)
        self.knownSurfaceDensityCoeff = self.measureBx(self.collisionForceTorque + self.externalForceTorque)
        self.knownSurfaceDensityCoeff += rho
        bCoeff = self.measureb(self.knownSurfaceDensityCoeff) #- self.measureAx(mu)

        toc = time.perf_counter()
        print("Solve for b took", toc - tic, "s")

        # solve Ax=b for x where x is the unknownSurfaceDensityCoeff
        # initial guess chosen from previous timestep
        numIters = 0
        def callback(_):
            nonlocal numIters
            numIters += 1

        tic = time.perf_counter()
        self.unknownSurfaceDensityCoeff, exitCode = gmres(A=self.measureAop, b=bCoeff, x0=self.unknownSurfaceDensityCoeff, callback=callback, tol=1e-6)
        assert not exitCode, "GMRES failed"
        toc = time.perf_counter()
        print("GMRES took ", toc - tic, "s")
        print("GMRES converged in", numIters, "iterations")

        # store results
        surfaceDensityCoeff = self.unknownSurfaceDensityCoeff + self.knownSurfaceDensityCoeff
        surfaceDensityGridVals = self.unknownSurfaceDensityGridVals + self.knownSurfaceDensityGridVals
        tic = time.perf_counter()
        self.postProcessAndStoreResults(surfaceDensityCoeff, surfaceDensityGridVals)
        toc = time.perf_counter()
        print("postProcessAndStoreResults took ", toc - tic, "s")

    def calcFlow(self, targetPos):
        """Calculate flow at targetPos induced by known and unknown surfaceDensityGridVals on the particle surfaces"""
        surfaceDensityGridVals = self.unknownSurfaceDensityGridVals + self.knownSurfaceDensityGridVals

        numTargetPts = len(targetPos)
        targetVel = np.zeros_like(targetPos)
        for i in range(numTargetPts):
            # target properties
            targetPosi = targetPos[i, :]

            for pidxSource in range(self.numPtcLocal):
                # particle properties
                ptcSource = self.particleContainer[pidxSource]
                layerSource = ptcSource.getLayer(self.name)
                numGridPtsSource = layerSource.getNumGridPts()
                ptsidxSource = self.particlePtsIndex[pidxSource]

                for j in range(numGridPtsSource):
                    gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                    
                    rVec = targetPosi - gridPointSourcej
                    velocityVec = self.stokeslet(rVec, surfaceDensityGridVals[ptsidxSource + j, :])
                    targetVel[i, :] += velocityVec * layerSource.getGridWeight(j)
        return targetVel

    def calcFlowAnalytical(self, targetPos, externalFlowY=0, externalFlowZ=1):
        """Calculate flow at targetPos induced by a prolate spheroid aligned with the z-axis"""
        numTargetPts = len(targetPos)
        targetVel = np.zeros_like(targetPos)
        for i in range(numTargetPts):
            # target properties MOM cancelled package 
            targetPosi = targetPos[i, :]
              
            u1, u2, u3, _ = calcAnalyticalFlowProlateSpheroid(U3=externalFlowZ, U2=externalFlowY, mu=self.runConfig.viscosity, 
                                a=self.runConfig.particleAxisLen3, b=self.runConfig.particleAxisLen2, 
                                x=[targetPosi[0]], y=[targetPosi[1]], z=[targetPosi[2]])
            targetVel[i, 0] = u1[0].real
            targetVel[i, 1] = u2[0].real
            targetVel[i, 2] = u3[0].real
        return targetVel

    def calcFlowAnalyticalSurface(self, targetPos, externalFlowY=0, externalFlowZ=1):
        """Calculate flow at targetPos (on the spheroid) induced by a prolate spheroid aligned with the z-axis"""
        numTargetPts = len(targetPos)
        targetVel = np.zeros_like(targetPos)
        for i in range(numTargetPts):
            # target properties MOM cancelled package 
            targetPosi = targetPos[i, :]
              
            u1, u2, u3, _ = calcAnalyticalFlowProlateSpheroid(U3=externalFlowZ, U2=externalFlowY, mu=self.runConfig.viscosity, 
                                a=self.runConfig.particleAxisLen3, b=self.runConfig.particleAxisLen2, 
                                x=[targetPosi[0]], y=[targetPosi[1]], z=[targetPosi[2]])
            targetVel[i, 0] = u1[0].real
            targetVel[i, 1] = u2[0].real
            targetVel[i, 2] = u3[0].real
        return targetVel

    def postProcessAndStoreResults(self, surfaceDensityCoeff, surfaceDensityGridVals):
        """Calculate surface grid velocity and centroid velocity. 
        Store grid velocity, centroid velocity, and surface density in particleContainer"""

        for pidxTarget in range(self.numPtcLocal):
            # particle properties
            ptcTarget = self.particleContainer[pidxTarget]
            layerTarget = ptcTarget.getLayer(self.name)
            numGridPtsTarget = layerTarget.getNumGridPts()
            numSpectralCoeffTarget = layerTarget.getNumSpectralCoeff()
            centroidTarget = layerTarget.getCentroidCurrentConfig(ptcTarget.pos)
            invMomentOfInertiaTensorTarget = layerTarget.getInvMomentOfInertiaTensorCurrentConfig()
            invSurfaceAreaTarget = 1 / layerTarget.getSurfaceArea()

            # single layer used to solve for surface velocity at each point
            vel = np.zeros(3)
            omega = np.zeros(3)
            surfaceVelGrid = np.zeros([3 * numGridPtsTarget])
            for i in range(numGridPtsTarget):
                gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)
                spherePointTargeti = layerTarget.getSpherePointRefConfig(i)

                for pidxSource in range(self.numPtcLocal):
                    # particle properties
                    ptcSource = self.particleContainer[pidxSource]
                    layerSource = ptcSource.getLayer(self.name)
                    numGridPtsSource = layerSource.getNumGridPts()
                    ptsidxSource = self.particlePtsIndex[pidxSource]

                    if pidxSource == pidxTarget:
                        # particle - self interaction

                        # step 1. Setup quadrature with the target point as "north"
                        spherePointsSourceRotated, gridPointsSourceRotated, surfaceWeightsSource = layerSource.rotatedQuadrature(ptcSource.pos, i)

                        # import matplotlib.pyplot as plt
                        # from mpl_toolkits.mplot3d import Axes3D
                        # fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        # ax.scatter(spherePointTargeti[0], spherePointTargeti[1], spherePointTargeti[2], s=40, c='k')
                        # ax.scatter(spherePointsSourceRotated[:, 0], spherePointsSourceRotated[:, 1], spherePointsSourceRotated[:, 2])
                        # ax.scatter(gridPointsSourceRotated[:, 0]-ptcSource.pos[0], gridPointsSourceRotated[:, 1]-ptcSource.pos[1], gridPointsSourceRotated[:, 2]-ptcSource.pos[2])
                        # max_range = np.array([spherePointsSourceRotated[:,0].max()-spherePointsSourceRotated[:,0].min(), spherePointsSourceRotated[:,1].max()-spherePointsSourceRotated[:,1].min(), spherePointsSourceRotated[:,2].max()-spherePointsSourceRotated[:,2].min()]).max() / 2.0
                        # mid_x = (spherePointsSourceRotated[:,0].max() + spherePointsSourceRotated[:,0].min()) * 0.5
                        # mid_y = (spherePointsSourceRotated[:,1].max() + spherePointsSourceRotated[:,1].min()) * 0.5
                        # mid_z = (spherePointsSourceRotated[:,2].max() + spherePointsSourceRotated[:,2].min()) * 0.5
                        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
                        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
                        # ax.set_zlim(mid_z - max_range, mid_z + max_range)
                        # plt.show()

                        # fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        # ax.scatter(gridPointTargeti[0], gridPointTargeti[1], gridPointTargeti[2], s=40, c='k')
                        # ax.scatter(gridPointsSourceRotated[:, 0], gridPointsSourceRotated[:, 1], gridPointsSourceRotated[:, 2])
                        # max_range = np.array([gridPointsSourceRotated[:,0].max()-gridPointsSourceRotated[:,0].min(), gridPointsSourceRotated[:,1].max()-gridPointsSourceRotated[:,1].min(), gridPointsSourceRotated[:,2].max()-gridPointsSourceRotated[:,2].min()]).max() / 2.0
                        # mid_x = (gridPointsSourceRotated[:,0].max() + gridPointsSourceRotated[:,0].min()) * 0.5
                        # mid_y = (gridPointsSourceRotated[:,1].max() + gridPointsSourceRotated[:,1].min()) * 0.5
                        # mid_z = (gridPointsSourceRotated[:,2].max() + gridPointsSourceRotated[:,2].min()) * 0.5
                        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
                        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
                        # ax.set_zlim(mid_z - max_range, mid_z + max_range)
                        # plt.show()


                        # step 2. Use hyperinterpolation to evaluate the grid quantities 
                        #           at the rotated quadrature points
                        coeffidx = self.particleCoeffIndex[pidxSource]
                        surfaceDensityCoeffSource = surfaceDensityCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeffTarget)].reshape([numSpectralCoeffTarget, 3])
                        surfaceDensityHyperinterpolated = layerSource.hyperinterpolateSurfaceVectorField(
                            surfaceDensityCoeffSource, spherePointsSourceRotated)

                        # step 3. Apply the addition theorem to the modified Stokeslet
                        for j in range(numGridPtsSource):
                            xHat = spherePointTargeti
                            yHat = spherePointsSourceRotated[j, :]
                            px = gridPointTargeti
                            py = gridPointsSourceRotated[j, :]

                            rVec = yHat - xHat
                            prVec = py - px
                            assert np.isclose(np.linalg.norm(rVec), np.linalg.norm(prVec))

                            if np.isclose(np.linalg.norm(rVec), 0.0):
                                # when r == 0, the weight is zero. Skipping is allowed
                                continue

                            R = np.linalg.norm(rVec) / np.linalg.norm(prVec)
                            MPhi = R * self.rstokeslet(prVec) @ surfaceDensityHyperinterpolated[j, :]
                        
                            alpha = 0
                            for l in range(layerSource.order + 1):
                                alpha += eval_legendre(l, np.dot(xHat, yHat))
                            surfaceVelGrid[3 * i + 0: 3 * i + 3] += alpha * MPhi * surfaceWeightsSource[j]
                    else:
                        # particle - particle interaction
                        for j in range(numGridPtsSource):
                            gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                            
                            rVec = gridPointTargeti - gridPointSourcej
                            surfaceDensitySourcej = surfaceDensityGridVals[3 * (ptsidxSource + j): 3 * (ptsidxSource + j) + 3]
                            velocityVec = self.stokeslet(rVec, surfaceDensitySourcej)
                            surfaceVelGrid[3 * i + 0: 3 * i + 3] += velocityVec * layerSource.getGridWeight(j)

                # calculate and store vel/omega at ptcTarget.pos
                surfaceVelGridi = surfaceVelGrid[3 * i + 0: 3 * i + 3]
                vel += surfaceVelGridi * layerTarget.getGridWeight(i)
                omega += np.cross(surfaceVelGridi, gridPointTargeti - centroidTarget) * layerTarget.getGridWeight(i) 

            # store vel/omega
            ptcTarget.vel = vel * invSurfaceAreaTarget
            ptcTarget.omega = invMomentOfInertiaTensorTarget @ omega
            print(ptcTarget.vel, ptcTarget.omega)
            
            direction = np.array([0.0, -1.0, 0.0])
            direction_perp = np.array([1.0, 0.0, 0.0])
            print("U dot dir", np.dot(ptcTarget.vel, direction))
            print("U dot dir_perp", np.dot(ptcTarget.vel, direction_perp))


            # store surfaceVelocityCoeff and surfaceDensityCoeff
            numSpectralCoeffTarget = layerTarget.getNumSpectralCoeff()
            coeffidxTarget = self.particleCoeffIndex[pidxTarget]
            layerTarget.surfaceDensityCoeff = surfaceDensityCoeff[3*coeffidxTarget:3*(coeffidxTarget + numSpectralCoeffTarget)]
            layerTarget.surfaceVelocityCoeff = layerTarget.decomposeSurfaceVectorFcn(surfaceVelGrid.reshape([numGridPtsTarget, 3])).flatten()


    # def postProcessAndStoreResults(self, surfaceDensityCoeff, surfaceDensityGridVals):
    #     """Calculate surface grid velocity and centroid velocity. 
    #     Store grid velocity, centroid velocity, and surface density in particleContainer"""

    #     for pidxTarget in range(self.numPtcLocal):
    #         # particle properties
    #         ptcTarget = self.particleContainer[pidxTarget]
    #         layerTarget = ptcTarget.getLayer(self.name)
    #         numGridPtsTarget = layerTarget.getNumGridPts()
    #         centroidTarget = layerTarget.getCentroidCurrentConfig(ptcTarget.pos)
    #         invMomentOfInertiaTensorTarget = layerTarget.getInvMomentOfInertiaTensorCurrentConfig()
    #         invSurfaceAreaTarget = 1 / layerTarget.getSurfaceArea()

    #         # single layer used to solve for suface velocity at each point
    #         # TODO: delete me
    #         uVecGrid = np.zeros([numGridPtsTarget, 3])
    #         vel2 = np.zeros(3)
    #         omega2 = np.zeros(3)

    #         vel = np.zeros(3)
    #         omega = np.zeros(3)
    #         surfaceVelGrid = np.zeros([3 * numGridPtsTarget])
    #         for i in range(numGridPtsTarget):
    #             gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)

    #             for pidxSource in range(self.numPtcLocal):
    #                 if pidxTarget == pidxSource:
    #                     selfInteraction = True
    #                 else:
    #                     selfInteraction = False

    #                 # particle properties
    #                 ptcSource = self.particleContainer[pidxSource]
    #                 layerSource = ptcSource.getLayer(self.name)
    #                 numGridPtsSource = layerSource.getNumGridPts()
    #                 ptsidxSource = self.particlePtsIndex[pidxSource]

    #                 integrand = np.zeros([numGridPtsSource, 3])
    #                 for j in range(numGridPtsSource):
    #                     # analytical integration proved we can skip the singular self-interaction component
    #                     if selfInteraction:
    #                         if i == j:
    #                             continue
    #                     gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                        
    #                     rVec = gridPointTargeti - gridPointSourcej
    #                     velocityVec = self.stokeslet(rVec, surfaceDensityGridVals[3 * (ptsidxSource + j): 3 * (ptsidxSource + j) + 3])
    #                     integrand[j,:] = velocityVec
    #                     surfaceVelGrid[3 * i: 3 * i + 3] += velocityVec * layerSource.getGridWeight(j)


    #                 tmpCoeff = layerSource.decomposeSurfaceVectorFcn(integrand.flatten())
    #                 integrand_recon = layerSource.reconstructSurfaceVectorFcn(tmpCoeff)
    #                 integrand_recon = integrand_recon.reshape([numGridPtsSource, 3])
    #                 import matplotlib.pyplot as plt
    #                 from mpl_toolkits.mplot3d import Axes3D
    #                 fig = plt.figure()
    #                 ax = fig.add_subplot(111, projection='3d')
    #                 xyz = layerTarget.sharedPS.gridPointsRefConfig.reshape([numGridPtsTarget,3]).T
    #                 # ax.quiver(xyz[0,:], xyz[1,:], xyz[2,:], integrand[:,0], integrand[:,1], integrand[:,2], color='red', length=2)                   
    #                 # ax.quiver(xyz[0,:], xyz[1,:], xyz[2,:], integrand_recon[:,0], integrand_recon[:,1], integrand_recon[:,2], color='blue', length=2)   

    #                 # ax.scatter(xyz[0,:], xyz[1,:], xyz[2,:], c=np.sqrt(np.sum(integrand*integrand,axis=1)))                
    #                 # max_range = np.array([xyz[0,:].max()-xyz[0,:].min(), xyz[1,:].max()-xyz[1,:].min(), xyz[2,:].max()-xyz[2,:].min()]).max() / 2.0
    #                 # mid_x = (xyz[0,:].max() + xyz[0,:].min()) * 0.5
    #                 # mid_y = (xyz[1,:].max() + xyz[1,:].min()) * 0.5
    #                 # mid_z = (xyz[2,:].max() + xyz[2,:].min()) * 0.5
    #                 # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #                 # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #                 # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #                 # ax.scatter(xyz[0,i], xyz[1,i], xyz[2,i], s=40, c='k')
    #                 # plt.show()
    #                 # print(integrand)


    #             # calculate and store vel/omega at ptcTarget.pos
    #             vel[0] += surfaceVelGrid[3 * i + 0] * layerTarget.getGridWeight(i)
    #             vel[1] += surfaceVelGrid[3 * i + 1] * layerTarget.getGridWeight(i)
    #             vel[2] += surfaceVelGrid[3 * i + 2] * layerTarget.getGridWeight(i)
    #             omega[0] += (surfaceVelGrid[3 * i + 1] * (gridPointTargeti[2] - centroidTarget[2])
    #                        - surfaceVelGrid[3 * i + 2] * (gridPointTargeti[1] - centroidTarget[1])) * layerTarget.getGridWeight(i)              
    #             omega[1] += (surfaceVelGrid[3 * i + 2] * (gridPointTargeti[0] - centroidTarget[0])
    #                        - surfaceVelGrid[3 * i + 0] * (gridPointTargeti[2] - centroidTarget[2])) * layerTarget.getGridWeight(i)   
    #             omega[2] += (surfaceVelGrid[3 * i + 0] * (gridPointTargeti[1] - centroidTarget[1])
    #                        - surfaceVelGrid[3 * i + 1] * (gridPointTargeti[0] - centroidTarget[0])) * layerTarget.getGridWeight(i)   

    #             # TODO: delete me
    #             x = [gridPointTargeti[0] - centroidTarget[0]]
    #             y = [gridPointTargeti[1] - centroidTarget[1]]
    #             z = [gridPointTargeti[2] - centroidTarget[2]]
    #             xyz = gridPointTargeti - centroidTarget
    #             # u1, u2, u3, _ = calcAnalyticalFlowProlateSpheroid(U3=1, U2=0, mu=self.runConfig.viscosity, 
    #             #     a=self.runConfig.particleAxisLen3, b=self.runConfig.particleAxisLen2, x=x, y=y, z=z)
    #             # u1, u2, u3, _ = calcAnalyticalFlowSphere(U=1, mu=self.runConfig.viscosity, 
    #             #     radius=self.runConfig.particleRadius, xyz=np.array([x,y,z]))
    #             uVec, _, _, _ = calcAnalyticalFlowMicroHydroTranslation(layerTarget.sharedPS.thetasA[i], layerTarget.sharedPS.phisA[i])
    #             # uVec, _ = calcAnalyticalFlowSphere(xyz, mu=self.runConfig.viscosity, a=self.runConfig.particleRadius)
    #             uVecGrid[i, :] = uVec
    #             vel2[0] += uVec[0] * layerTarget.getGridWeight(i)
    #             vel2[1] += uVec[1] * layerTarget.getGridWeight(i)
    #             vel2[2] += uVec[2] * layerTarget.getGridWeight(i)
    #             omega2[0] += (uVec[1] * (gridPointTargeti[2] - centroidTarget[2])
    #                       - uVec[2] * (gridPointTargeti[1] - centroidTarget[1])) * layerTarget.getGridWeight(i)              
    #             omega2[1] += (uVec[2] * (gridPointTargeti[0] - centroidTarget[0])
    #                       - uVec[0] * (gridPointTargeti[2] - centroidTarget[2])) * layerTarget.getGridWeight(i)   
    #             omega2[2] += (uVec[0] * (gridPointTargeti[1] - centroidTarget[1])
    #                       - uVec[1] * (gridPointTargeti[0] - centroidTarget[0])) * layerTarget.getGridWeight(i)   


    #         # store vel/omega
    #         ptcTarget.vel = vel * invSurfaceAreaTarget
    #         ptcTarget.omega = invMomentOfInertiaTensorTarget @ omega
    #         print(ptcTarget.vel, ptcTarget.omega)
            
    #         # store surfaceVelocityCoeff and surfaceDensityCoeff
    #         numSpectralCoeffTarget = layerTarget.getNumSpectralCoeff()
    #         coeffidxTarget = self.particleCoeffIndex[pidxTarget]
    #         layerTarget.surfaceDensityCoeff = surfaceDensityCoeff[3*coeffidxTarget:3*(coeffidxTarget + numSpectralCoeffTarget)]
    #         layerTarget.surfaceVelocityCoeff = layerTarget.decomposeSurfaceVectorFcn(surfaceVelGrid)

    #         #TODO: delete
    #         # surfaceDensityGridVals = layerTarget.reconstructSurfaceVectorFcn(layerTarget.surfaceDensityCoeff)
    #         # surfaceVelocityGridVals = layerTarget.reconstructSurfaceVectorFcn(layerTarget.surfaceVelocityCoeff)
    #         # np.sum(surfaceDensityGridVals * surfaceVelocityGridVals * np.repeat(layerTarget.gridWeights, 3))
    #         # np.sum(surfaceDensityGridVals * surfaceVelGrid * np.repeat(layerTarget.gridWeights, 3))
    #         # np.sum(surfaceDensityGridVals * np.repeat(layerTarget.gridWeights, 3))


    #         surfaceRelVelGrid = surfaceVelGrid.reshape([numGridPtsTarget, 3])
    #         diff = uVecGrid - surfaceRelVelGrid
    #         # import matplotlib.pyplot as plt
    #         # from mpl_toolkits.mplot3d import Axes3D
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(111, projection='3d')
    #         # # xyz = np.array([np.sin(layerTarget.thetasA) * np.cos(layerTarget.phisA), np.sin(layerTarget.thetasA) * np.sin(layerTarget.phisA), np.cos(layerTarget.thetasA)])
    #         # xyz = layerTarget.sharedPS.gridPointsRefConfig.reshape([numGridPtsTarget,3]).T
    #         # ax.quiver(xyz[0,:], xyz[1,:], xyz[2,:], uVecGrid[:,0], uVecGrid[:,1], uVecGrid[:,2], color='red', length=0.2)
    #         # ax.quiver(xyz[0,:], xyz[1,:], xyz[2,:], surfaceRelVelGrid[:,0], surfaceRelVelGrid[:,1], surfaceRelVelGrid[:,2], color='blue', length=0.2)
    #         # # ax.quiver(xyz[0,:], xyz[1,:], xyz[2,:], diff[:,0], diff[:,1], diff[:,2], color='green', length=2)
    #         # #TODO: how to incorperate viscocity? 
            
    #         # max_range = np.array([xyz[0,:].max()-xyz[0,:].min(), xyz[1,:].max()-xyz[1,:].min(), xyz[2,:].max()-xyz[2,:].min()]).max() / 2.0
    #         # mid_x = (xyz[0,:].max()+xyz[0,:].min()) * 0.5
    #         # mid_y = (xyz[1,:].max()+xyz[1,:].min()) * 0.5
    #         # mid_z = (xyz[2,:].max()+xyz[2,:].min()) * 0.5
    #         # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #         # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #         # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #         # plt.show()

    #         # print("L2norm:", layerTarget.sharedPS.l2normSurfaceVecField((uVecGrid - surfaceRelVelGrid).flatten()) / layerTarget.sharedPS.l2normSurfaceVecField(uVecGrid.flatten()))
    #         # print("")

    def getUserInputSurfaceDensity(self):
        # TODO: update such that the prescribedSurfaceDensityCoeff is stored for each particle
        prescribedSurfaceDensityCoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128)
        if os.path.exists(self.runConfig.prescribedSurfaceDensityFile):
            for pidx in range(self.numPtcLocal):     
                # particle properties
                ptc = self.particleContainer[pidx]
                layer = ptc.getLayer(self.name)
                numSpectralCoeff = layer.getNumSpectralCoeff()
                coeffidx = self.particleCoeffIndex[pidx]

                # get and store the prescribed coeffs
                prescribedSurfaceDensityCoeffi = layer.getUserInputSurfaceDensity(self.runConfig.prescribedSurfaceDensityFile)
                prescribedSurfaceDensityCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)] = prescribedSurfaceDensityCoeffi
        return prescribedSurfaceDensityCoeff
                
    def getSurfaceDensityCoeffFromParticleContainer(self):
        surfaceDensityCoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128)
        for pidx in range(self.numPtcLocal):     
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]

            surfaceDensityCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)] = layer.surfaceDensityCoeff.flatten()
        return surfaceDensityCoeff

    def measureAx(self, unknownSurfaceDensityCoeff):
        """measure A operating on unknownSurfaceDensityCoeff"""
        measureAxCoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128) 
        # step 1: convert knownSurfaceDensityCoeff from spectral to grid
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)

            # reconstruct surfaceDensity from surfaceDensityCoeff
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]
            
            tic = time.perf_counter()
            coeffVSH = unknownSurfaceDensityCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)].reshape([numSpectralCoeff, 3])
            unknownSurfaceDensityGridVals = layer.reconstructSurfaceVectorFcn(coeffVSH).flatten()
            toc = time.perf_counter()
            print("reconstructSurfaceVectorFcn took", toc - tic, "s")

            # store the values
            numGridPts = layer.getNumGridPts()
            pointidx = self.particlePtsIndex[pidx]            
            self.unknownSurfaceDensityGridVals[3*pointidx:3*(pointidx + numGridPts)] = unknownSurfaceDensityGridVals

        # step 2: compute the operations on the grid values
        tic = time.perf_counter()
        AxGridVals = self.Ax(self.unknownSurfaceDensityGridVals)
        toc = time.perf_counter()
        print("Ax took", toc - tic, "s")

        # step 3: convert from grid to spectral
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)

            # convert the grid values to spectral coeff
            numGridPts = layer.getNumGridPts()
            pointidx = self.particlePtsIndex[pidx]   
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]

            tic = time.perf_counter()
            surfaceVectorFcn = AxGridVals[3*pointidx:3*(pointidx + numGridPts)].reshape([numGridPts, 3])
            measureAxCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)] = layer.decomposeSurfaceVectorFcn(
                surfaceVectorFcn).flatten()
            toc = time.perf_counter()
            print("decomposeSurfaceVectorFcn took", toc - tic, "s")
        return measureAxCoeff


    def measureb(self, knownSurfaceDensityCoeff):
        """measure b, which is solely a function of knownSurfaceDensityCoeff"""
        bCoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128) 
        # step 1: convert knownSurfaceDensityCoeff from spectral to grid
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)

            # reconstruct surfaceDensity from surfaceDensityCoeff
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]
            coeffVSH = knownSurfaceDensityCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)].reshape([numSpectralCoeff, 3])
            knownSurfaceDensityGridVals = layer.reconstructSurfaceVectorFcn(coeffVSH).flatten()

            # store the values
            numGridPts = layer.getNumGridPts()
            pointidx = self.particlePtsIndex[pidx]            
            self.knownSurfaceDensityGridVals[3*pointidx:3*(pointidx + numGridPts)] = knownSurfaceDensityGridVals

        # step 2: compute the operations on the grid values
        bGridVals = self.b(self.knownSurfaceDensityGridVals)

        # step 3: convert from grid to spectral
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)

            # convert the grid values to spectral coeff
            numGridPts = layer.getNumGridPts()
            pointidx = self.particlePtsIndex[pidx]   
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]
            surfaceVectorFcn = bGridVals[3*pointidx:3*(pointidx + numGridPts)].reshape([numGridPts, 3])
            bCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)] = layer.decomposeSurfaceVectorFcn(
                surfaceVectorFcn).flatten()
        return bCoeff


    def measureBx(self, totalForceTorque):
        VSHcoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128)
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)
            centroid = layer.getCentroidCurrentConfig(ptc.pos)
            invMomentOfInertiaTensor = layer.getInvMomentOfInertiaTensorCurrentConfig()
            numGridPts = layer.getNumGridPts()
            invSurfaceArea = 1 / layer.getSurfaceArea()

            # calculate the total force and torque induced by surfaceDensity
            totalForce = totalForceTorque[6 * pidx + 0: 6 * pidx + 3]
            totalTorque = totalForceTorque[6 * pidx + 3: 6 * pidx + 6]

            # distribute the total force and torque to the entire surface
            tauInvT = invMomentOfInertiaTensor @ totalTorque
            densityFromTotalForceAndTorque = np.zeros(3 * numGridPts)
            for j in range(numGridPts):
                gridPoint = layer.getGridPointCurrentConfig(ptc.pos, j)
                densityFromTotalForceAndTorque[3 * j + 0: 3 * j + 3] += totalForce * invSurfaceArea
                densityFromTotalForceAndTorque[3 * j + 0: 3 * j + 3] += np.cross(tauInvT, gridPoint - centroid)

            # measure densityFromInducedTotalForceAndTorque using VSH 
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]
            surfaceVectorFcn = densityFromTotalForceAndTorque.reshape([numGridPts, 3])
            VSHcoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)] = layer.decomposeSurfaceVectorFcn(surfaceVectorFcn).flatten()

        return VSHcoeff     


    def measureLx(self, surfaceDensityCoeff):
        """Measure the 6 dimensional nullspace operator applied to surfaceDensityCoeff
        calculates and then measures the surface density from the induced total force and torque
        """

        VSHcoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128)
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)
            centroid = layer.getCentroidCurrentConfig(ptc.pos)
            invMomentOfInertiaTensor = layer.getInvMomentOfInertiaTensorCurrentConfig()
            numGridPts = layer.getNumGridPts()
            invSurfaceArea = 1 / layer.getSurfaceArea()

            # reconstruct surfaceDensity from surfaceDensityCoeff
            numSpectralCoeff = layer.getNumSpectralCoeff()
            coeffidx = self.particleCoeffIndex[pidx]
            coeffVSH = surfaceDensityCoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)].reshape([numSpectralCoeff, 3])
            surfaceDensity = layer.reconstructSurfaceVectorFcn(coeffVSH).flatten()

            # calculate the total force and torque induced by surfaceDensity
            inducedTotalForce = np.zeros(3)
            inducedTotalTorque = np.zeros(3)
            for j in range(numGridPts):
                gridPoint = layer.getGridPointCurrentConfig(ptc.pos, j)
                surfaceDensityj = surfaceDensity[3 * j + 0: 3 * j + 3]
                inducedTotalForce += surfaceDensityj * layer.getGridWeight(j)
                inducedTotalTorque += np.cross(gridPoint - centroid, surfaceDensityj) * layer.getGridWeight(j)
            
            # distribute the total force and torque to the entire surface
            inducedTotalTorque = invMomentOfInertiaTensor @ inducedTotalTorque
            densityFromInducedTotalForceAndTorque = np.zeros(3 * numGridPts)
            for j in range(numGridPts):
                gridPoint = layer.getGridPointCurrentConfig(ptc.pos, j)

                densityFromInducedTotalForceAndTorque[3 * j + 0: 3 * j + 3] += inducedTotalForce * invSurfaceArea
                densityFromInducedTotalForceAndTorque[3 * j + 0: 3 * j + 3] += np.cross(inducedTotalTorque, gridPoint - centroid)

            # measure densityFromInducedTotalForceAndTorque using VSH 
            surfaceVectorFcn = densityFromInducedTotalForceAndTorque.reshape([numGridPts, 3])
            VSHcoeff[3*coeffidx:3*(coeffidx + numSpectralCoeff)] = layer.decomposeSurfaceVectorFcn(surfaceVectorFcn).flatten()

        return VSHcoeff 

    # TODO: account for periodicity within rVec
    # TODO: delete this operator
    def measureSx(self, surfaceDensityCoeff):
        """Measure the dense single layer hydrodynamic interaction operator applied to surfaceDensityCoeff"""

        VSHcoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128)
        for pidxTarget in range(self.numPtcLocal):
            # particle properties
            ptcTarget = self.particleContainer[pidxTarget]
            layerTarget = ptcTarget.getLayer(self.name)
            numGridPtsTarget = layerTarget.getNumGridPts()

            STarget = np.zeros([3 * numGridPtsTarget])
            for i in range(numGridPtsTarget):
                gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)

                for pidxSource in range(self.numPtcLocal):
                    if pidxTarget == pidxSource:
                        selfInteraction = True
                    else:
                        selfInteraction = False

                    # particle properties
                    ptcSource = self.particleContainer[pidxSource]
                    layerSource = ptcSource.getLayer(self.name)
                    numGridPtsSource = layerSource.getNumGridPts()

                    # reconstruct surfaceDensity from surfaceDensityCoeff
                    numSpectralCoeffSource = layerSource.getNumSpectralCoeff()
                    coeffidxSource = self.particleCoeffIndex[pidxSource]
                    surfaceDensitySource = layerSource.reconstructSurfaceVectorFcn(surfaceDensityCoeff[3*coeffidxSource:3*(coeffidxSource + numSpectralCoeffSource)])

                    for j in range(numGridPtsSource):
                        # analytical integration proved we can skip the singular self-interaction component
                        if selfInteraction:
                            if i == j:
                                continue
                        gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                        
                        rVec = gridPointTargeti - gridPointSourcej
                        surfaceDensitySourcej = surfaceDensitySource[3 * j: 3 * j + 3]
                        velocityVec = self.stokeslet(rVec, surfaceDensitySourcej)
                        STarget[3 * i: 3 * i + 3] += velocityVec * layerSource.getGridWeight(j)
        
            # measure STarget
            numSpectralCoeffTarget = layerTarget.getNumSpectralCoeff()
            coeffidxTarget = self.particleCoeffIndex[pidxTarget]
            VSHcoeff[3*coeffidxTarget:3*(coeffidxTarget + numSpectralCoeffTarget)] = layerTarget.decomposeSurfaceVectorFcn(STarget)

        return VSHcoeff  

    # TODO: account for periodicity within rVec (best wait for FMM to do this for us)
    def measureJx(self, surfaceDensityCoeff):
        """Measure the dense hydrodynamic interaction operator applied to surfaceDensityCoeff with singularity subtraction"""

        VSHcoeff = np.zeros(3 * self.particleCoeffIndex[-1], dtype=np.complex128)
        for pidxTarget in range(self.numPtcLocal):
            # particle properties
            ptcTarget = self.particleContainer[pidxTarget]
            layerTarget = ptcTarget.getLayer(self.name)
            numGridPtsTarget = layerTarget.getNumGridPts()

            # reconstruct surfaceDensity from surfaceDensityCoeff
            numSpectralCoeffTarget = layerTarget.getNumSpectralCoeff()
            coeffidxTarget = self.particleCoeffIndex[pidxTarget]
            gridDensityTarget = layerTarget.reconstructSurfaceVectorFcn(surfaceDensityCoeff[3*coeffidxTarget:3*(coeffidxTarget + numSpectralCoeffTarget)])

            JTarget = np.zeros([3 * numGridPtsTarget])
            for i in range(numGridPtsTarget):
                gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)
                gridNormTargeti = layerTarget.getGridNormCurrentConfig(i)
                gridDensityTargeti = gridDensityTarget[3 * i: 3 * i + 3]

                for pidxSource in range(self.numPtcLocal):
                    if pidxTarget == pidxSource:
                        selfInteraction = True
                    else:
                        selfInteraction = False

                    # particle properties
                    ptcSource = self.particleContainer[pidxSource]
                    layerSource = ptcSource.getLayer(self.name)
                    numGridPtsSource = layerSource.getNumGridPts()

                    # reconstruct surfaceDensity from surfaceDensityCoeff
                    numSpectralCoeffSource = layerSource.getNumSpectralCoeff()
                    coeffidxSource = self.particleCoeffIndex[pidxSource]
                    gridDensitySource = layerSource.reconstructSurfaceVectorFcn(surfaceDensityCoeff[3*coeffidxSource:3*(coeffidxSource + numSpectralCoeffSource)])

                    for j in range(numGridPtsSource):
                        # singular subtraction allows us to skip the singular self-interaction component
                        if selfInteraction:
                            if i == j:
                                continue
                        gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                        gridNormSourcej = layerSource.getGridNormCurrentConfig(j)
                        gridDensitySourcej = gridDensitySource[3 * j: 3 * j + 3]

                        rVec = gridPointTargeti - gridPointSourcej
                        tractionVec = self.stresslet(rVec, gridDensityTargeti, gridDensitySourcej, gridNormTargeti, gridNormSourcej)
                        JTarget[3 * i: 3 * i + 3] += tractionVec * layerSource.getGridWeight(j)
        
            # measure JTarget
            numSpectralCoeffTarget = layerTarget.getNumSpectralCoeff()
            coeffidxTarget = self.particleCoeffIndex[pidxTarget]
            surfaceVectorFcn = JTarget.reshape([numGridPtsTarget, 3])
            VSHcoeff[3*coeffidxTarget:3*(coeffidxTarget + numSpectralCoeffTarget)] = layerTarget.decomposeSurfaceVectorFcn(surfaceVectorFcn).flatten()

        return VSHcoeff  

    def stokeslet(self, rVec, forceVec):
        def kdelta(i, j):
            return 1 if i == j else 0

        coeff = 1 / (8 * np.pi)
        rNorm = np.linalg.norm(rVec)
        rNormInvScaled = 1 / rNorm * coeff
        rNormInv3Scaled = 1 / rNorm**3 * coeff

        velocityVec = np.zeros(3)
        for i in range(3):
            for j in range(3):
                velocityVec[i] +=  (kdelta(i, j) * rNormInvScaled + rNormInv3Scaled * rVec[i] * rVec[j]) * forceVec[j]
        return velocityVec

    def rstokeslet(self, rVec):
        def kdelta(i, j):
            return 1 if i == j else 0

        S = np.zeros([3, 3])
        coeff = 1 / (8 * np.pi)

        rNorm = np.linalg.norm(rVec)
        if not np.isclose(rNorm, 0.0):
            rNormInv2 = 1 / rNorm**2

            for i in range(3):
                for j in range(3):
                    S[i, j] =  coeff * (rNormInv2 * rVec[i] * rVec[j] + kdelta(i, j))
        else: 
            for i in range(3):
                for j in range(3):
                    S[i, j] =  coeff * kdelta(i, j)

        return S

    def stresslet(self, rVec, forceVecTarget, forceVecSource, normalVecTarget, normalVecSource):
        coeff = - 3 / (4 * np.pi)
        rNorm = np.linalg.norm(rVec)
        rNormInv5Scaled = 1 / rNorm**5 * coeff

        tractionVec = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tractionVec[i] +=  rNormInv5Scaled * rVec[i] * rVec[j] * rVec[k] * (forceVecSource[j] * normalVecTarget[k] + forceVecTarget[k] * normalVecSource[j])
        return tractionVec

    def Ax(self, unknownSurfaceDensityGridVals):
        """A operating on unknownSurfaceDensity"""
        tic = time.perf_counter()
        out = self.Lx(unknownSurfaceDensityGridVals)
        toc = time.perf_counter()
        print("Lx took", toc - tic, "s")

        tic = time.perf_counter()
        # out += self.Jx(unknownSurfaceDensityGridVals)
        out += self.JMatrix @ unknownSurfaceDensityGridVals
        toc = time.perf_counter()
        print("Jx took", toc - tic, "s")
        return out


    def b(self, knownSurfaceDensityGridVals):
        """b, which is solely a function of totalForce and totalTorque"""
        return -self.Jx(knownSurfaceDensityGridVals) 
       

    def Bx(self, totalForceTorque):
        result = np.zeros(3 * self.particlePtsIndex[-1])
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)
            numGridPts = layer.getNumGridPts()
            ptsidx = self.particlePtsIndex[pidx]
            centroid = layer.getCentroidCurrentConfig(ptc.pos)
            invMomentOfInertiaTensor = layer.getInvMomentOfInertiaTensorCurrentConfig()
            invSurfaceArea = 1 / layer.getSurfaceArea()

            # calculate the total force and torque induced by surfaceDensity
            totalForce = totalForceTorque[6 * pidx + 0: 6 * pidx + 3]
            totalTorque = totalForceTorque[6 * pidx + 3: 6 * pidx + 6]

            # distribute the total force and torque to the entire surface
            tauInvT = invMomentOfInertiaTensor @ totalTorque
            densityFromTotalForceAndTorque = np.zeros(3 * numGridPts)
            for j in range(numGridPts):
                gridPoint = layer.getGridPointCurrentConfig(ptc.pos, j)
                
                densityFromTotalForceAndTorque[3 * j + 0: 3 * j + 3] += totalForce * invSurfaceArea
                densityFromTotalForceAndTorque[3 * j + 0: 3 * j + 3] += np.cross(tauInvT, gridPoint - centroid)

            # store the results
            result[3*ptsidx:3*(ptsidx + numGridPts)] = densityFromTotalForceAndTorque
        return result     


    def Lx(self, surfaceDensityGridVals):
        """6 dimensional nullspace operator applied to surfaceDensityGridVals
        calculates the surface density from the induced total force and torque
        """

        result = np.zeros(3 * self.particlePtsIndex[-1])
        for pidx in range(self.numPtcLocal):
            # particle properties
            ptc = self.particleContainer[pidx]
            layer = ptc.getLayer(self.name)
            numGridPts = layer.getNumGridPts()
            ptsidx = self.particlePtsIndex[pidx]
            centroid = layer.getCentroidCurrentConfig(ptc.pos)
            invMomentOfInertiaTensor = layer.getInvMomentOfInertiaTensorCurrentConfig()
            invSurfaceArea = 1 / layer.getSurfaceArea()

            # calculate the total force and torque induced by surfaceDensityGridVals
            inducedTotalForce = np.zeros(3)
            inducedTotalTorque = np.zeros(3)
            for j in range(numGridPts):
                gridPoint = layer.getGridPointCurrentConfig(ptc.pos, j)
                
                inducedTotalForce[0] += surfaceDensityGridVals[3 * (ptsidx + j) + 0] * layer.getGridWeight(j)
                inducedTotalForce[1] += surfaceDensityGridVals[3 * (ptsidx + j) + 1] * layer.getGridWeight(j)
                inducedTotalForce[2] += surfaceDensityGridVals[3 * (ptsidx + j) + 2] * layer.getGridWeight(j)
                inducedTotalTorque[0] += ((gridPoint[1] - centroid[1]) * surfaceDensityGridVals[3 * (ptsidx + j) + 2] 
                                        - (gridPoint[2] - centroid[2]) * surfaceDensityGridVals[3 * (ptsidx + j) + 1]) * layer.getGridWeight(j)
                inducedTotalTorque[1] += ((gridPoint[2] - centroid[2]) * surfaceDensityGridVals[3 * (ptsidx + j) + 0] 
                                        - (gridPoint[0] - centroid[0]) * surfaceDensityGridVals[3 * (ptsidx + j) + 2]) * layer.getGridWeight(j)
                inducedTotalTorque[2] += ((gridPoint[0] - centroid[0]) * surfaceDensityGridVals[3 * (ptsidx + j) + 1] 
                                        - (gridPoint[1] - centroid[1]) * surfaceDensityGridVals[3 * (ptsidx + j) + 0]) * layer.getGridWeight(j)
            
            # distribute the total force and torque to the entire surface
            inducedTotalTorque = invMomentOfInertiaTensor @ inducedTotalTorque
            densityFromInducedTotalForceAndTorque = np.zeros(3 * numGridPts)
            for j in range(numGridPts):
                gridPoint = layer.getGridPointCurrentConfig(ptc.pos, j)

                densityFromInducedTotalForceAndTorque[3 * j + 0] += inducedTotalForce[0] * invSurfaceArea
                densityFromInducedTotalForceAndTorque[3 * j + 1] += inducedTotalForce[1] * invSurfaceArea
                densityFromInducedTotalForceAndTorque[3 * j + 2] += inducedTotalForce[2] * invSurfaceArea
                densityFromInducedTotalForceAndTorque[3 * j + 0] += (inducedTotalTorque[1] * (gridPoint[2] - centroid[2])
                                                                  - inducedTotalTorque[2] * (gridPoint[1] - centroid[1]))              
                densityFromInducedTotalForceAndTorque[3 * j + 1] += (inducedTotalTorque[2] * (gridPoint[0] - centroid[0])
                                                                  - inducedTotalTorque[0] * (gridPoint[2] - centroid[2]))  
                densityFromInducedTotalForceAndTorque[3 * j + 2] += (inducedTotalTorque[0] * (gridPoint[1] - centroid[1])
                                                                  - inducedTotalTorque[1] * (gridPoint[0] - centroid[0]))  

            # store the results
            result[3*ptsidx:3*(ptsidx + numGridPts)] = densityFromInducedTotalForceAndTorque
        return result   

    def Sx(self, surfaceDensity):
        """Dense single layer hydrodynamic interaction operator applied to surfaceDensity"""

        result = np.zeros(3 * self.particlePtsIndex[-1])
        for pidxTarget in range(self.numPtcLocal):
            # particle properties
            ptcTarget = self.particleContainer[pidxTarget]
            layerTarget = ptcTarget.getLayer(self.name)
            numGridPtsTarget = layerTarget.getNumGridPts()
            ptsidxTarget = self.particlePtsIndex[pidxTarget]

            STarget = np.zeros(3 * numGridPtsTarget)
            for i in range(numGridPtsTarget):
                gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)
                
                for pidxSource in range(self.numPtcLocal):
                    if pidxTarget == pidxSource:
                        selfInteraction = True
                    else:
                        selfInteraction = False

                    # particle properties
                    ptcSource = self.particleContainer[pidxSource]
                    layerSource = ptcSource.getLayer(self.name)
                    numGridPtsSource = layerSource.getNumGridPts()
                    ptsidxSource = self.particlePtsIndex[pidxSource]

                    for j in range(numGridPtsSource):
                        # analytical integration proved we can skip the singular self-interaction component
                        if selfInteraction:
                            if i == j:
                                continue

                        gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                        
                        rVec = gridPointTargeti - gridPointSourcej
                        velocityVec = self.stokeslet(rVec, surfaceDensity[3 * (ptsidxSource + j): 3 * (ptsidxSource + j) + 3])
                        STarget[3 * i: 3 * i + 3] += velocityVec * layerSource.getGridWeight(j)
        
            # store the results
            result[3*ptsidxTarget:3*(ptsidxTarget + numGridPtsTarget)] = STarget
        return result   


    def Jx(self, surfaceDensity):
        """Dense hydrodynamic interaction operator applied to surfaceDensity with singularity subtraction"""

        result = np.zeros(3 * self.particlePtsIndex[-1])
        for pidxTarget in range(self.numPtcLocal):
            # particle properties
            ptcTarget = self.particleContainer[pidxTarget]
            layerTarget = ptcTarget.getLayer(self.name)
            numGridPtsTarget = layerTarget.getNumGridPts()
            ptsidxTarget = self.particlePtsIndex[pidxTarget]

            JTarget = np.zeros(3 * numGridPtsTarget)
            for i in range(numGridPtsTarget):
                gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)
                gridNormTargeti = layerTarget.getGridNormCurrentConfig(i)
                gridDensityTargeti = surfaceDensity[3 * (ptsidxTarget + i): 3 * (ptsidxTarget + i) + 3]

                for pidxSource in range(self.numPtcLocal):
                    if pidxTarget == pidxSource:
                        selfInteraction = True
                    else:
                        selfInteraction = False

                    # particle properties
                    ptcSource = self.particleContainer[pidxSource]
                    layerSource = ptcSource.getLayer(self.name)
                    numGridPtsSource = layerSource.getNumGridPts()
                    ptsidxSource = self.particlePtsIndex[pidxSource]

                    for j in range(numGridPtsSource):
                        # singular subtraction allows us to skip the singular self-interaction component
                        if selfInteraction:
                            if i == j:
                                continue
                        gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                        gridNormSourcej = layerSource.getGridNormCurrentConfig(j)
                        gridDensitySourcej = surfaceDensity[3 * (ptsidxSource + j): 3 * (ptsidxSource + j) + 3]

                        rVec = gridPointTargeti - gridPointSourcej
                        tractionVec = self.stresslet(rVec, gridDensityTargeti, gridDensitySourcej, gridNormTargeti, gridNormSourcej)
                        JTarget[3 * i: 3 * i + 3] += tractionVec * layerSource.getGridWeight(j)
        
            # store the results
            result[3*ptsidxTarget:3*(ptsidxTarget + numGridPtsTarget)] = JTarget
        return result   






    def calcJMatrix(self):
        """Dense hydrodynamic interaction matrix"""

        J = np.zeros([3 * self.particlePtsIndex[-1], 3 * self.particlePtsIndex[-1]])
        for pidxTarget in range(self.numPtcLocal):
            # particle properties
            ptcTarget = self.particleContainer[pidxTarget]
            layerTarget = ptcTarget.getLayer(self.name)
            numGridPtsTarget = layerTarget.getNumGridPts()
            ptsidxTarget = self.particlePtsIndex[pidxTarget]

            for i in range(numGridPtsTarget):
                gridPointTargeti = layerTarget.getGridPointCurrentConfig(ptcTarget.pos, i)
                gridNormTargeti = layerTarget.getGridNormCurrentConfig(i)

                for pidxSource in range(self.numPtcLocal):
                    if pidxTarget == pidxSource:
                        selfInteraction = True
                    else:
                        selfInteraction = False

                    # particle properties
                    ptcSource = self.particleContainer[pidxSource]
                    layerSource = ptcSource.getLayer(self.name)
                    numGridPtsSource = layerSource.getNumGridPts()
                    ptsidxSource = self.particlePtsIndex[pidxSource]

                    for j in range(numGridPtsSource):
                        # singular subtraction allows us to skip the singular self-interaction component
                        if selfInteraction:
                            if i == j:
                                continue
                        gridPointSourcej = layerSource.getGridPointCurrentConfig(ptcSource.pos, j)
                        gridNormSourcej = layerSource.getGridNormCurrentConfig(j)

                        rVec = gridPointTargeti - gridPointSourcej
                        tractionTensor = self.calcTractionTensor(rVec)
                        A = np.tensordot(tractionTensor, gridNormSourcej, (1,0)) * layerSource.getGridWeight(j)
                        B = np.tensordot(tractionTensor, gridNormTargeti, (2,0)) * layerSource.getGridWeight(j)
                        J[3 * ptsidxTarget + 3 * i: 3 * ptsidxTarget + 3 * i + 3, 
                          3 * ptsidxTarget + 3 * i: 3 * ptsidxTarget + 3 * i + 3] += A
                        J[3 * ptsidxTarget + 3 * i: 3 * ptsidxTarget + 3 * i + 3, 
                          3 * ptsidxSource + 3 * j: 3 * ptsidxSource + 3 * j + 3] += B
        return J   

    def calcTractionTensor(self, rVec):
        coeff = - 3 / (4 * np.pi)
        rNorm = np.linalg.norm(rVec)
        rNormInv5Scaled = 1 / rNorm**5 * coeff

        tractionTensor = np.zeros([3, 3, 3])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tractionTensor[i,j,k] =  rNormInv5Scaled * rVec[i] * rVec[j] * rVec[k]
        return tractionTensor










                                   
    def dumpMTX(self, fileNum):
        """Dump A, x, and b to matrix market files. Warning this requires N evaluations of Aop"""
        N = 3 * self.particleCoeffIndex[-1]

        # loop over the columns of I to compute A from Aop
        Adense = np.zeros([N, N])
        I = np.identity(N)
        for c in range(N):
            Adense[:, c] = self.measureAop.matvec(I[:, c])
        print("A size: " + str(N) + "x" + str(N))
        print("A Rank:", np.linalg.matrix_rank(Adense))
        print("A Cond Number:", np.linalg.cond(Adense))

        # dump A, x, and b
        mmwrite("./result/A_" + str(fileNum) + ".mtx", Adense)
        mmwrite("./result/xCoeff_" + str(fileNum) + ".mtx", self.xCoeff.reshape(-1,1))
        mmwrite("./result/bCoeff_" + str(fileNum) + ".mtx", self.bCoeff.reshape(-1,1))