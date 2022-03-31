#include "Particle.hpp"
#include "Util/Base64.hpp"

/*****************************************************
 *  Genus zero particle
 *****************************************************/

template <int maxSpectralDegree>
Particle<maxSpectralDegree>::Particle(const int &gid, const double &radius, const double &radiusCollision, const double &length,
                   const double &lengthCollision, const double pos[3], const double orientation[4]) {
    this->gid = gid;
    this->radius = radius;
    this->radiusCollision = radiusCollision;
    this->length = length;
    this->lengthCollision = lengthCollision;
    if (pos == nullptr) {
        Emap3(this->pos).setZero();
    } else {
        for (int i = 0; i < 3; i++) {
            this->pos[i] = pos[i];
        }
    }
    if (orientation == nullptr) {
        Emapq(this->orientation).setIdentity();
    } else {
        for (int i = 0; i < 4; i++) {
            this->orientation[i] = orientation[i];
        }
    }

    clear();
    return;
}

template <int maxSpectralDegree>
void Particle<maxSpectralDegree>::clear() {
    Emap3(vel).setZero();
    Emap3(omega).setZero();
    Emap3(velCol).setZero();
    Emap3(omegaCol).setZero();
    Emap3(velBi).setZero();
    Emap3(omegaBi).setZero();
    Emap3(velNonB).setZero();
    Emap3(omegaNonB).setZero();

    Emap3(force).setZero();
    Emap3(torque).setZero();
    Emap3(forceCol).setZero();
    Emap3(torqueCol).setZero();
    Emap3(forceBi).setZero();
    Emap3(torqueBi).setZero();
    Emap3(forceNonB).setZero();
    Emap3(torqueNonB).setZero();

    Emap3(velBrown).setZero();
    Emap3(omegaBrown).setZero();

    sepmin = std::numeric_limits<double>::max();
    globalIndex = GEO_INVALID_INDEX;
    rank = -1;
}

template <int maxSpectralDegree>
void Particle<maxSpectralDegree>::dumpParticle() const {
    printf("gid %d, R %g, RCol %g, L %g, LCol %g, pos %g, %g, %g\n", gid, radius, radiusCollision, length,
           lengthCollision, pos[0], pos[1], pos[2]);
    printf("vel %g, %g, %g; omega %g, %g, %g\n", vel[0], vel[1], vel[2], omega[0], omega[1], omega[2]);
    printf("orient %g, %g, %g, %g\n", orientation[0], orientation[1], orientation[2], orientation[3]);
}

template <int maxSpectralDegree>
void Particle<maxSpectralDegree>::stepEuler(double dt) {
    Emap3(pos) += Emap3(vel) * dt;
    Equatn currOrient = Emapq(orientation);
    EquatnHelper::rotateEquatn(currOrient, Emap3(omega), dt);
    Emapq(orientation).x() = currOrient.x();
    Emapq(orientation).y() = currOrient.y();
    Emapq(orientation).z() = currOrient.z();
    Emapq(orientation).w() = currOrient.w();
}

template <int maxSpectralDegree>
void Particle<maxSpectralDegree>::writeAscii(FILE *fptr) const {
    Evec3 direction = ECmapq(orientation) * Evec3(0, 0, 1);
    Evec3 minus = ECmap3(pos) - 0.5 * length * direction;
    Evec3 plus = ECmap3(pos) + 0.5 * length * direction;
    char typeChar = isImmovable ? 'S' : 'C';
    fprintf(fptr, "%c %d %.8g %.8g %.8g %.8g %.8g %.8g %.8g %d\n", //
            typeChar, gid, radius,                                 //
            minus[0], minus[1], minus[2], plus[0], plus[1], plus[2], group);
}
