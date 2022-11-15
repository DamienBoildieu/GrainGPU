#include "kernels.cuh"
#include "sphParameters.cuh"

using namespace utils;
namespace sph {
//*******************************************************************************
DEVICE uint32 start(ConstPtr<uint32> cellsStop, int32 cell) 
{
	return cell ? cellsStop[cell-1] : 0;
}
//*******************************************************************************
DEVICE uint32 stop(ConstPtr<uint32> cellsStop, int32 cell)
{
	return cellsStop[cell];
}
//*****************************************************************************
GLOBAL
void getNeighbors(Ptr<uint32> nbNeighbor, ConstPtr<Vec2f> pos, ConstPtr<uint32> cellsStop,
    ConstPtr<uint32> worldKey, Vec2<uint32> dim, ConstPtr<Vec2f> lowerLimits, uint32 partNumber,
    uint32 worldBegin, float maxRadius)
{
	const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32 gridSize = blockDim.x * gridDim.x;
    const float dx = maxRadius;
    const float dy = maxRadius;
	const int nbcx = 2; //from cpu version
	const int nbcy = 2;
	for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
		const int world = worldKey[tid]-worldBegin;
		const int gridBegin = world*dim.x()*dim.y();
		const Vec2f lower{lowerLimits[world]};
		//get the computed particle
		const int idx = min(dim.x()-1,
			max(0,(int) floor((pos[tid].x()-lower.x())/dx)));
		const int idy = min(dim.y()-1,
		  	max(0,(int) floor((pos[tid].y()-lower.y())/dy)));

		int nbCell = 0;
		int nbNeigh = 0;
		for(int i=idx-nbcx; i<=idx+nbcx; i++){
			for(int j=idy-nbcy; j<=idy+nbcy; j++){
				if(i>=0 && j>=0 && i<dim.x() && j<dim.y() ){
					const int cell = gridBegin + i + j*dim.x();
					if(cell<((world+1)*dim.x()*dim.y())){
						nbCell++;
						//printf("Cell %d : start : %d - stop : %d \n", cell, start(cellsStop, cell), stop(cellsStop, cell));
						for (uint32 partId=start(cellsStop, cell); partId<stop(cellsStop, cell); partId++) {
							/*if (partId==tid)
								continue;*/
							//printf("tid : %d -- partId : %d \n", tid, partId);
							const double d2 = (pos[tid]-pos[partId]).norm2();
                            if (d2>maxRadius*maxRadius)
								continue;
							//printf("part %d : %f ; %f -- neigh %d : %f ; %f -- d2 %f -- d %f\n", tid, params.pos[tid].x(), params.pos[tid].y(), 
							//	partId, params.pos[partId].x(), params.pos[partId].y(), d2, sqrt(d2));
							nbNeigh++;
						}
					}
				}
			}
		}
		nbNeighbor[tid] = nbNeigh;
		//printf("pos : %f ; %f -- cell %d ; %d -- nbCell : %d  -- nbNeigh : %d \n", pos[tid].x(), pos[tid].y(), idx, idy, nbCell, nbNeigh);
	}	
}
//*****************************************************************************
GLOBAL 
void createGrid(ConstPtr<Vec2f> pos, ConstPtr<uint32> worldKey,	Ptr<uint32> indexes,
    Ptr<uint32> offset, Ptr<uint32> partByCell, uint32 partNumber, uint32 worldBegin,
    Vec2<uint32> dim, ConstPtr<Vec2f> lowerLimits, float maxRadius)
{
    const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32 gridSize = blockDim.x * gridDim.x;
	const float dx = maxRadius;
	const float dy = maxRadius;
	for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
		const uint32 world = worldKey[tid]-worldBegin;
		const Vec2f lower{lowerLimits[world]};
		const Vec2f posTid = pos[tid];
		const int32 idx = min(dim.x()-1,
			max(0,(int32) floor((posTid.x()-lower.x())/dx)));
		const int32 idy = min(dim.y()-1,
			max(0,(int32) floor((posTid.y()-lower.y())/dy)));
		const uint32 index = world*dim.x()*dim.y()+idx+idy*dim.x();
		indexes[tid] = index;
		offset[tid] = atomicAdd(&partByCell[index], 1u);
	}
}
//*****************************************************************************
GLOBAL 
void computeRhoP(ConstPtr<Vec2f> pos, ConstPtr<float> mass, ConstPtr<float> radius,
	Ptr<float> rho, Ptr<float> p, ConstPtr<uint32> cellsStop, ConstPtr<uint32> worldKey,
	Vec2<uint32> dim, ConstPtr<Vec2f> lowerLimits, uint32 partNumber, uint32 worldBegin,
    float maxRadius, float rho0, float taitsB)
{
	const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32 gridSize = blockDim.x * gridDim.x;

	const float dx = maxRadius;
	const float dy = maxRadius;
	const int32 nbcx = 2; //from cpu version
	const int32 nbcy = 2;
	for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {	//get the computed particle and init rho
		const int32 world = worldKey[tid]-worldBegin;
		const int32 gridBegin = world*dim.x()*dim.y();
		const Vec2f lower{lowerLimits[world]};
		const Vec2f posTid = pos[tid];
		const float massTid = mass[tid];
		const float radiusTid = radius[tid];

		const int32 idx = min(dim.x()-1,
			max(0,(int32) floor((posTid.x()-lower.x())/dx)));
		const int32 idy = min(dim.y()-1,
		  	max(0,(int32) floor((posTid.y()-lower.y())/dy)));
		float rhoTmp = 0;

		for(int32 i=idx-nbcx; i<=idx+nbcx; i++){
			for(int32 j=idy-nbcy; j<=idy+nbcy; j++){
				if(i>=0 && j>=0 && i<dim.x() && j<dim.y() ){
					const int32 cell = gridBegin + i + j*dim.x();
						
					if(cell<((world+1)*dim.x()*dim.y())){
						for (uint32 partId=start(cellsStop, cell); partId<stop(cellsStop, cell); partId++) {
							if (partId==tid)
								continue;
                            const float h = max(radiusTid, radius[partId]);
							//compute rho
							const float d2 = (posTid-pos[partId]).norm2();
							//if (world==8)
							//	printf("radiusTid: %f -- radiusOther : %f\n", radiusTid, radius[partId], d2);
                            if (d2>=h*h)
								continue;
							//if (world==5)
							//	printf("mass : %f -- h2 : %f -- d2 : %f -- world : %u\n", massTid, h*h, d2, world);
                            rhoTmp += massTid * 315. *
                                pow(h*h - d2, 3.)/(64.*pi*pow(h,9.));
						}
					}
				}
			}
		}
		rho[tid] = rhoTmp;
		//compute p
		p[tid] = (pow(rhoTmp/rho0,7.)-1.) * taitsB;
	}	
}
//*****************************************************************************
GLOBAL 
void computeForces(ConstPtr<Vec2f> pos, ConstPtr<float> mass, ConstPtr<float> radius,
	ConstPtr<float> rho, ConstPtr<float> p, Ptr<Vec2d> force, ConstPtr<Vec2f> vel,
	ConstPtr<Vec2d> externalForces, Ptr<uint32> cellsStop, ConstPtr<uint32> worldKey,
	Vec2<uint32> dim, ConstPtr<Vec2f> lowerLimits, uint32 partNumber, uint32 worldBegin,
	float maxRadius, float mu, Ptr<Vec2d> fP, Ptr<Vec2d> fV, Ptr<Vec2d> fE)
{
	const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32 gridSize = blockDim.x * gridDim.x;
	
	const float dx = maxRadius;
	const float dy = maxRadius;
	const int32 nbcx = 2; //from cpu version
	const int32 nbcy = 2;
	for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
		if (rho[tid]==0.)
			continue;

		const int32 world = worldKey[tid]-worldBegin;
		const int32 gridBegin = world*dim.x()*dim.y();
		const Vec2f lower{lowerLimits[world]};
		const Vec2d externalForce{externalForces[world]};
		const Vec2f posTid = pos[tid];
		const float radiusTid = radius[tid];
		const Vec2f velTid = vel[tid];
		const float pTid = p[tid];
		const float rhoTid = rho[tid];

		const int32 idx = min(dim.x()-1,
			max(0,(int32) floor((posTid.x()-lower.x())/dx)));
		const int32 idy = min(dim.y()-1,
			max(0,(int32) floor((posTid.y()-lower.y())/dy)));
	
		Vec2d fPP{0.};
		Vec2d fVV{0.};
		for(int32 i=idx-nbcx; i<=idx+nbcx; i++){
			for(int32 j=idy-nbcy; j<=idy+nbcy; j++){
				if(i>=0 && j>=0 && i<dim.x() && j<dim.y() ){
					const int32 cell = gridBegin + i + j*dim.x();
						
					if(cell<((world+1)*dim.x()*dim.y())){
						for (uint32 partId=start(cellsStop, cell); partId<stop(cellsStop, cell); partId++) {
							if (partId==tid || rho[partId]==0.)
								continue;
                            const float pV2 = mass[partId]/rho[partId];
                            const float h = max(radiusTid, radius[partId]);
                            Vec2f p1p2 = posTid-pos[partId];
							if (p1p2.x()==0. && p1p2.y()==0.)
								continue;
							const float d = p1p2.normalize();
                            if (d>h)
								continue;
							Vec2d WP{p1p2.x(), p1p2.y()};
                            fPP += (-45.*pow(h-d,2)/(pi*pow(h,6)))
								*0.5*pV2*(pTid+p[partId])*WP;
						
                            double Wv = (45.*(h-d)/(pi*pow(h,6)))*pV2;
                            fVV.x() += (vel[partId].x()-velTid.x()) * Wv;
                            fVV.y() += (vel[partId].y()-velTid.y()) * Wv;
						}
					}
				}
			}
		}
		const Vec2d fEE = externalForce*rhoTid;
		fVV *= double(mu);
		if (fP!=nullptr)
			fP[tid] = -fPP;
		if (fV!=nullptr)
        	fV[tid] = fVV;
		if (fE!=nullptr)
			fE[tid] = fEE;
        force[tid] = fPP + fVV + fEE;
		//force[tid] = -fP + (double)mu*fV + externalForce*rho[tid]*externalForce.norm();
	}
}
//*****************************************************************************
GLOBAL 
void integrate(Ptr<Vec2f> pos, ConstPtr<float> rho, Ptr<Vec2f> vel,
    ConstPtr<Vec2d> force, float dt, uint32 partNumber)
{
	const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32 gridSize = blockDim.x * gridDim.x;
	for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
		if (rho[tid]==0.)
			continue;
		vel[tid].x() += force[tid].x()/rho[tid]*dt;
		vel[tid].y() += force[tid].y()/rho[tid]*dt;
		pos[tid] += vel[tid]*dt;		
	}
}
//*****************************************************************************
GLOBAL
void collision(Ptr<Vec2f> pos, Ptr<Vec2f> vel, ConstPtr<uint32> worldKey,
	ConstPtr<Vec2f> lowerLimits, ConstPtr<Vec2f> upperLimits, float elast, float fric,
	uint32 partNumber, uint32 worldBegin)
{
	const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32 gridSize = blockDim.x * gridDim.x;
	for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
		const uint32 world = worldKey[tid]-worldBegin;
		const Vec2f lower{lowerLimits[world]};
		const Vec2f upper{upperLimits[world]};
		const Vec2f posTid = pos[tid];
		bool collide = false;
		Vec2f N{0.};
		if(posTid.x()<lower.x()) {
			pos[tid].x() = lower.x(); 
			collide = true;
			N = {1., 0.};
		} else if(posTid.x()>upper.x()) {
			pos[tid].x() = upper.x();
			collide = true;
			N = {-1., 0.};
		}
		if(posTid.y()<lower.y()) {
			pos[tid].y() = lower.y();
			collide = true;
			N = {0., 1.};
		} 
		if(posTid.y()>upper.y()) { 
			pos[tid].y() = upper.y();
			collide = true;
			N = {0., -1.};
		}
		
		if(collide)
		{
			const Vec2f velCpy = vel[tid];
			const Vec2f vN = velCpy.dot(N)*N;
			const Vec2f vT = velCpy - vN;
			vel[tid] = fric*vT-elast*vN;
		}
	}
}
//*****************************************************************************
GLOBAL 
void updateMass(utils::Ptr<float> mass, utils::Ptr<float> radius, float avg,
	float dAvg, float dStdev, float rho0, uint32 partNumber)
{
	const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 gridSize = blockDim.x * gridDim.x;
    for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
        const float oldMass = mass[tid];
        const float dMass = oldMass - avg;
        const float newMass = oldMass + dMass*dStdev - dMass + dAvg;
        mass[tid] = newMass;
        radius[tid] = sph::radius(newMass, rho0);
    }
}
}
