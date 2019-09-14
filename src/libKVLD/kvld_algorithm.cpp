/** @basic structures implementation
 ** @author Zhe Liu
 **/

/*
Copyright (C) 2011-12 Zhe Liu and Pierre Moulon.
All rights reserved.

This file is part of the KVLD library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "kvld_algorithm.h"

//template<typename T>
IntegralImages::IntegralImages(const Image<float>& I){
		map.Resize(I.Width()+1,I.Height()+1);
		map.fill(0);
		for (size_t y=0;y<I.Height();y++)
			for (size_t x=0;x<I.Width();x++){
				map(y+1,x+1)=double(I(y,x))+map(y,x+1)+map(y+1,x)-map(y,x);
			}
	}

float getRange(const Image<float>& I,int a,const float p, const float ratio){
  float range=ratio*sqrt(float(3*I.Height()*I.Width())/(p*a*PI));
  std::cout<<"range ="<<range<<std::endl;
  return range;
}

//=============================IO interface, convertion of object types======================//

std::ofstream& writeDetector(std::ofstream& out, const keypoint& feature){
  out<<feature.x<<" "<<feature.y<<" "<<feature.scale<<" "<<feature.angle<<std::endl;
  /*for(int i=0;i<128;i++)  
    out<<feature.vec[i]<<" ";
  out<<std::endl;*/
return out;
}

std::ifstream& readDetector(std::ifstream& in,keypoint& point){
  in>>point.x>>point.y>>point.scale>>point.angle;
  //for(int i=0;i<128;i++)  {
  //  in>>point.vec[i];
  //}
return in;
}
