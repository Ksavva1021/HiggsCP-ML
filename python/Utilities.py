import ROOT
from PolarimetricA1 import PolarimetricA1
import math
import vector

def Boost(p,boost):
   '''
   Function to boost a particle's four-vector using another four-vector
   '''   
   
   boost = vector.obj(px=boost.Px(),py=boost.Py(),pz=boost.Pz())
   p = vector.obj(px=p.Px(),py=p.Py(),pz=p.Pz(),E=p.E())
   p = p.boost(boost)
   return ROOT.TLorentzVector(p.px,p.py,p.pz,p.E)

def PolarimetricVector(charged_pions, neutral_pions, tau):
   '''
   Function to calculate the polarimetric vector for a tau decay
   '''
   
   output = ROOT.TVector3(0,0,0)
   boost = tau.Fourvector.BoostVector()
   
   # Case 1 : One charged Pion , Zero Neutral Pions
   if len(charged_pions) == 1 and len(neutral_pions) == 0:
      pi = charged_pions[0].Fourvector

      pi = Boost(pi,-boost)

      output = pi.Vect()

   # Case 2 : One charged Pion , One Neutral Pion
   elif len(charged_pions) == 1 and len(neutral_pions) == 1:
      pi = charged_pions[0].Fourvector
      pi0 = neutral_pions[0].Fourvector
      tau = tau.Fourvector
      
      pi = Boost(pi,-boost)
      pi0 = Boost(pi0,-boost)
      tau = Boost(tau,-boost)

      q = pi - pi0
      P = tau
      N = tau - pi - pi0

      output = P.M()*(2*(q*N)*q.Vect() - q.Mag2()*N.Vect()) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P))) 
      
   # Case 3 : Three charged Pions , Zero Neutral Pions      
   elif len(charged_pions) == 3 and len(neutral_pions) == 0:
      TauA1andProd = []
      
      if charged_pions[1].charge == charged_pions[2].charge:
         TauA1andProd.append(tau.Fourvector)
         TauA1andProd.append(charged_pions[0].Fourvector)
         TauA1andProd.append(charged_pions[1].Fourvector)
         TauA1andProd.append(charged_pions[2].Fourvector)

      elif charged_pions[0].charge == charged_pions[2].charge:
         TauA1andProd.append(tau.Fourvector)
         TauA1andProd.append(charged_pions[1].Fourvector)
         TauA1andProd.append(charged_pions[0].Fourvector)
         TauA1andProd.append(charged_pions[2].Fourvector)        

      elif charged_pions[0].charge == charged_pions[1].charge:
         TauA1andProd.append(tau.Fourvector)
         TauA1andProd.append(charged_pions[2].Fourvector)
         TauA1andProd.append(charged_pions[0].Fourvector)
         TauA1andProd.append(charged_pions[1].Fourvector) 

      for i in range(len(TauA1andProd)):
         TauA1andProd[i] = Boost(TauA1andProd[i],-boost)

      a1pol = PolarimetricA1(TauA1andProd, taucharge=tau.charge)
      l_out = -a1pol.PVC()
      l_out = Boost(l_out,boost)
      return l_out
  
   l_output = ROOT.TLorentzVector(output,0.) 
   l_output = Boost(l_output,boost)
   
   return l_output

def YRho(charged_pion,neutral_pion,boost):
  '''
  Function to calculate Y_Rho variable for rho resonances
  '''
  
  pi = Boost(charged_pion,-boost)
  pi0 = Boost(neutral_pion,-boost)
  
  E_pi = pi.E()
  E_pi0 = pi0.E()
  y = (E_pi - E_pi0)/(E_pi + E_pi0)
 
  return y 


def AcoplanarityAngle_PV(p1, p2, p3, p4):
   '''
   Function to calculate the acoplanarity angle using Polarimetric Vectors
   p1 = PV1
   p2 = PV2
   p3 = tau1
   p4 = tau2
   '''
   
   boost = (p3+p4).BoostVector()
   p1 = Boost(p1,-boost)
   p2 = Boost(p2,-boost)
   p3 = Boost(p3,-boost)
   p4 = Boost(p4,-boost)

   h1 = p1.Vect().Unit()
   h2 = p2.Vect().Unit()   
   
   n1 = p3.Vect().Unit()
   n2 = p4.Vect().Unit()   

   k1 = h1.Cross(n1).Unit()
   k2 = h2.Cross(n2).Unit()

   angle = math.acos(k1.Dot(k2))
   sign = h1.Cross(h2).Dot(n1)
   if(sign<0): angle = 2*math.pi - angle

   return angle

def AcoplanarityAngle(p1,p2,p3,p4,type1,type2):
   '''
   Function to calculate the acoplanarity angle using Impact Parameter and/or Neutral Pion methods
   p1 = IP 1 or neutral pion 1
   p2 = IP2 or neutral pion 2
   p3 = charged pion 1
   p4 = charged pion 2
   '''
   
   boost = (p3+p4).BoostVector()
   
   # If the type contains "IP" (impact parameter), convert to unit vector
   if "IP" in type1: p1 = ROOT.TLorentzVector(p1.Vect().Unit(),0.)
   if "IP" in type2: p2 = ROOT.TLorentzVector(p2.Vect().Unit(),0.)

   p1 = Boost(p1,-boost)
   p2 = Boost(p2,-boost)
   p3 = Boost(p3,-boost)
   p4 = Boost(p4,-boost)

   # Calculate the plane normals
   n1 = p1.Vect() - p1.Vect().Dot(p3.Vect().Unit())*p3.Vect().Unit()    
   n2 = p2.Vect() - p2.Vect().Dot(p4.Vect().Unit())*p4.Vect().Unit()

   n1 = n1.Unit()
   n2 = n2.Unit()
    
   # Calculate the acoplanarity angle 
   angle = math.acos(n1.Dot(n2))
   sign = p4.Vect().Unit().Dot(n1.Cross(n2))   
   if(sign<0): angle = 2*math.pi - angle
   
   cp_sign = 1
   # If the type contains "NP" (neutral pion), calculate Y_Rho and modify angle
   if "NP" in type1:
      cp_sign *= YRho(p3,p1,ROOT.TVector3())
   if "NP" in type2:
      cp_sign *= YRho(p4,p2,ROOT.TVector3())

   # If the type contains "NP" and the Y_Rho sign is negative, modify the angle
   if "NP" in type1 or "NP" in type2:
      if(cp_sign<0):
         if(angle < math.pi):
            angle += math.pi
         else:
            angle -= math.pi

   return angle

