#include "ParticleShapes.hpp"
#include "Util/EigenDef.hpp"

#include <math.h>

constexpr double pi = 3.1415926535897932384626433;

/******************
 *                *
 * ParticleShapes *
 *                *
 ******************/

Evec3 ParticleShapes::calcSurfacePointA(const double thetaA, const double phiA) const {
    return Evec3(std::sin(thetaA) * std::cos(phiA), std::sin(thetaA) * std::sin(phiA), std::cos(thetaA));
}

Evec3 ParticleShapes::calcSurfacePointB(const double thetaB, const double phiB) const {
    return Evec3(std::cos(thetaB), std::sin(thetaB) * std::sin(phiB), std::sin(thetaB) * std::cos(phiB));
}

Emat3 ParticleShapes::calcPushforwardMatrix(const double thetaA, const double phiA, const double thetaB,
                                            const double phiB) const {
    const Evec3 spherePoint = calcSurfacePointA(thetaA, phiA);
    return calcPushforwardMatrix(spherePoint);
}

Emat3 ParticleShapes::calcPullbackMatrix(const double thetaA, const double phiA, const double thetaB,
                                         const double phiB) const {
    const Evec3 spherePoint = calcSurfacePointA(thetaA, phiA);
    return calcPullbackMatrix(spherePoint);
}

Evec3 ParticleShapes::calcPSphereVectorHatPThetaA(const double thetaA, const double phiA) const {
    return Evec3(std::cos(thetaA) * std::cos(phiA), std::cos(thetaA) * std::sin(phiA), -std::sin(thetaA));
}

Evec3 ParticleShapes::calcPSphereVectorHatPThetaB(const double thetaB, const double phiB) const {
    return Evec3(-std::sin(thetaB), std::cos(thetaB) * std::sin(phiB), std::cos(thetaB) * std::cos(phiB));
}

Evec3 ParticleShapes::calcPSphereVectorHatPPhiA(const double thetaA, const double phiA) const {
    return Evec3(-std::sin(thetaA) * std::sin(phiA), std::sin(thetaA) * std::cos(phiA), 0.0);
}

Evec3 ParticleShapes::calcPSphereVectorHatPPhiB(const double thetaB, const double phiB) const {
    return Evec3(0.0, std::sin(thetaB) * std::cos(phiB), -std::sin(thetaB) * std::sin(phiB));
}

void ParticleShapes::calcSurfaceNormalHatAndWeight(const double thetaA, const double phiA, const double thetaB,
                                                   const double phiB, const double sphereWeight, Evec *surfaceNormal,
                                                   double *surfaceWeight) const {
    // compute in a non-degenerate chart
    if (0.8 * pi / 4 < thetaA && thetaA <= 0.8 * pi) {
        const Evec3 thetaVec = calcPSphereVectorHatPThetaA(thetaA, phiA);
        const Evec3 phiVec = calcPSphereVectorHatPPhiA(thetaA, phiA);

        const Emat3 pushforwardMatrix = calcPushforwardMatrix(thetaA, phiA, thetaB, phiB);
        const Evec3 thetaVecPushed = pushforwardMatrix * thetaVec;
        const Evec3 phiVecPushed = pushforwardMatrix * phiVec;

        surfaceNormal = thetaVecPushed.cross(phiVecPushed);
        const double surfaceNormalNorm = np.linalg.norm(surfaceNormal);
        surfaceNormal /= surfaceNormalNorm;
        surfaceWeight = surfaceNormalNorm * sphereWeight / np.sin(thetaA);
    } else {
        const Evec3 thetaVec = calcPSphereVectorHatPThetaB(thetaB, phiB);
        const Evec3 phiVec = calcPSphereVectorHatPPhiB(thetaB, phiB);

        const Emat3 pushforwardMatrix = calcPushforwardMatrix(thetaA, phiA, thetaB, phiB);
        const Evec3 thetaVecPushed = pushforwardMatrix * thetaVec;
        const Evec3 phiVecPushed = pushforwardMatrix * phiVec;

        surfaceNormal = thetaVecPushed.cross(phiVecPushed);
        const double surfaceNormalNorm = np.linalg.norm(surfaceNormal);
        surfaceNormal /= surfaceNormalNorm;
        surfaceWeight = surfaceNormalNorm * sphereWeight / np.sin(thetaB);
    }
}

void ParticleShapes::calcThetaPhiAFromPoint(const Evec3 point, double *thetaA, double *phiA) const {
    thetaA = std::atan2(std::sqrt(std::pow(point[0], 2) + std::pow(point[1], 2)), point[2]);
    phiA = std::atan2(point[1], point[0]);
}

void ParticleShapes::calcThetaPhiBFromPoint(const Evec3 point, double *thetaB, double *phiB) const {
    thetaB = std::atan2(std::sqrt(std::pow(point[1], 2) + std::pow(point[2], 2)), point[0]);
    phiB = std::atan2(point[1], point[2]);
}

/**********
 *        *
 * Sphere *
 *        *
 **********/
Sphere::Sphere(const double radius) { radius = radius; }

void Sphere::initialize(const YAML::Node &config) { readConfig(config, VARNAME(radius), radius, ""); }

double Sphere::calcRadiusA(const double thetaA, const double phiA) const { return radius; }

double Sphere::calcRadiusB(const double thetaB, const double phiB) const { return radius; }

Evec3 Sphere::pushforwardPoint(const Evec3 spherePoint) const {
    const Evec3 surfacePoint = radius * spherePoint;
    return surfacePoint;
}

Emat3 Sphere::calcPushforwardMatrix(const Evec3 spherePoint) const {
    const Emat3 P = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
    return P;
}

Emat3 Sphere::calcPullbackMatrix(const Evec3 spherePoint) const {
    const Emat3 Pinv = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
    return Pinv;
}

void Sphere::echo() const {
    printf("------------------------\n");
    printf("Sphere\n");
    printf("radius: %g\n", radius);
    printf("------------------------\n");
}

/************
 *          *
 * Spheroid *
 *          *
 ************/
Spheroid::Spheroid(const double axisLenX, const double axisLenY, const double axisLenZ) {
    this->axisLenX = axisLenX;
    this->axisLenY = axisLenY;
    this->axisLenZ = axisLenZ;
}

void Spheroid::initialize(const YAML::Node &config) {
    readConfig(config, VARNAME(axisLenX), axisLenX, "");
    readConfig(config, VARNAME(axisLenY), axisLenY, "");
    readConfig(config, VARNAME(axisLenZ), axisLenZ, "");
}

double Spheroid::calcRadiusA(const double thetaA, const double phiA) const {
    const double xSphere = std::sin(thetaA) * std::cos(phiA);
    const double ySphere = std::sin(thetaA) * std::sin(phiA);
    const double zSphere = std::cos(thetaA);
    const double xSurface = axisLenX * xSphere;
    const double ySurface = axisLenY * ySphere;
    const double zSurface = axisLenZ * zSphere;

    return std::sqrt(std::pow(xSurface, 2) + std::pow(ySurface, 2) + std::pow(zSurface, 2));
}

double Spheroid::calcRadiusB(const double thetaB, const double phiB) const {
    const double xSphere = std::cos(thetaB);
    const double ySphere = std::sin(thetaB) * std::sin(phiB);
    const double zSphere = std::sin(thetaB) * std::cos(phiB);
    const double xSurface = axisLenX * xSphere;
    const double ySurface = axisLenY * ySphere;
    const double zSurface = axisLenZ * zSphere;

    return std::sqrt(std::pow(xSurface, 2) + std::pow(ySurface, 2) + std::pow(zSurface, 2));
}

Evec3 Spheroid::pushforwardPoint(const Evec3 spherePoint) const {
    const Evec3 surfacePoint = {axisLenX * spherePoint[0], axisLenY * spherePoint[1], axisLenZ * spherePoint[2]};
    return surfacePoint;
}

Emat3 Spheroid::calcPushforwardMatrix(const Evec3 spherePoint) const {
    const Emat3 P = {{axisLenX, 0., 0.}, {0., axisLenY, 0.}, {0., 0., axisLenZ}};
    return P;
}

Emat3 Spheroid::calcPullbackMatrix(const Evec3 spherePoint) const {
    const Emat3 Pinv = {{1. / axisLenX, 0., 0.}, {0., 1. / axisLenY, 0.}, {0., 0., 1. / axisLenZ}};
    return Pinv;
}

void Spheroid::echo() const {
    printf("------------------------\n");
    printf("Spheroid\n");
    printf("axisLenX: %g\n", axisLenX);
    printf("axisLenY: %g\n", axisLenY);
    printf("axisLenZ: %g\n", axisLenZ);
    printf("------------------------\n");
}

/*********
 *       *
 * Helix *
 *       *
 *********/
Helix::Helix(const double height, const double majorRadius, const double minorRadius, const double frequency) {
    this->height = height;
    this->majorRadius = majorRadius;
    this->minorRadius = minorRadius;
    this->frequency = frequency;
}

void Helix::initialize(const YAML::Node &config) {
    readConfig(config, VARNAME(height), height, "");
    readConfig(config, VARNAME(majorRadius), majorRadius, "");
    readConfig(config, VARNAME(minorRadius), minorRadius, "");
    readConfig(config, VARNAME(frequency), frequency, "");
}

double Helix::calcRadiusA(const double thetaA, const double phiA) const {
    const double xSphere = std::sin(thetaA) * std::cos(phiA);
    const double ySphere = std::sin(thetaA) * std::sin(phiA);
    const double zSphere = std::cos(thetaA);
    const double xSurface = minorRadius * xSphere + majorRadius * std::sin(zSphere * frequency);
    const double ySurface = minorRadius * ySphere + majorRadius * std::cos(zSphere * frequency);
    const double zSurface = zSphere * height;

    return std::sqrt(std::pow(xSurface, 2) + std::pow(ySurface, 2) + std::pow(zSurface, 2));
}

double Helix::calcRadiusB(const double thetaB, const double phiB) const {
    const double xSphere = std::cos(thetaB);
    const double ySphere = std::sin(thetaB) * std::sin(phiB);
    const double zSphere = std::sin(thetaB) * std::cos(phiB);
    const double xSurface = minorRadius * xSphere + majorRadius * std::sin(zSphere * frequency);
    const double ySurface = minorRadius * ySphere + majorRadius * std::cos(zSphere * frequency);
    const double zSurface = zSphere * height;

    return std::sqrt(std::pow(xSurface, 2) + std::pow(ySurface, 2) + std::pow(zSurface, 2));
}

Evec3 Helix::pushforwardPoint(const Evec3 spherePoint) const {
    const double xSphere = spherePoint[0];
    const double ySphere = spherePoint[1];
    const double zSphere = spherePoint[2];
    const double xPushedforwad = minorRadius * xSphere + majorRadius * std::sin(zSphere * frequency);
    const double yPushedforwad = minorRadius * ySphere + majorRadius * std::cos(zSphere * frequency);
    const double zPushedforwad = zSphere * height;
    const Evec3 surfacePoint = {xPushedforwad, yPushedforwad, zPushedforwad};
    return surfacePoint;
}

Emat3 Helix::calcPushforwardMatrix(const Evec3 spherePoint) const {
    const double zSphere = spherePoint[2];
    Emat3 P = {{minorRadius, 0., majorRadius * frequency * std::cos(zSphere * frequency)},
               {0., minorRadius, -majorRadius * frequency * std::sin(zSphere * frequency)},
               {0., 0., height}};
    return P;
}

Emat3 Helix::calcPullbackMatrix(const Evec3 spherePoint) const {
    const Emat3 P = calcPushforwardMatrix(spherePoint);
    return P.inverse();
}

void Helix::echo() const {
    printf("------------------------\n");
    printf("Helix\n");
    printf("height: %g\n", height);
    printf("majorRadius: %g\n", majorRadius);
    printf("minorRadius: %g\n", minorRadius);
    printf("frequency: %g\n", frequency);
    printf("------------------------\n");
}