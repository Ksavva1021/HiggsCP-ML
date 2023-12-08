import numpy as np
from typing import List
import ROOT
import math

class PolarimetricA1:
   def __init__(self, TauA1andProd, ReferenceFrame=None, taucharge=None):
      if len(TauA1andProd) != 4:
          print("Warning!! Size of a1 input vector != 4 !!")
      if ReferenceFrame is None:
          ReferenceFrame = TauA1andProd[0]
      if taucharge is None:
          taucharge = 1
      self.Setup(TauA1andProd, ReferenceFrame, taucharge)

   def Setup(self, TauA1andProd, ReferenceFrame, taucharge):
      self.mpi = 0.13957018  # GeV
      self.mpi0 = 0.1349766  # GeV
      self.mtau = 1.776  # GeV
      self.coscab = 0.975
      self.mrho = 0.773  # GeV
      self.mrhoprime = 1.370  # GeV
      self.ma1 = 1.251  # GeV
      self.mpiprime = 1.300  # GeV
      self.Gamma0rho = 0.145  # GeV
      self.Gamma0rhoprime = 0.510  # GeV
      self.Gamma0a1 = 0.599  # GeV
      self.Gamma0piprime = 0.3  # GeV
      self.fpi = 0.093  # GeV
      self.fpiprime = 0.08  # GeV
      self.gpiprimerhopi = 5.8  # GeV
      self.grhopipi = 6.08  # GeV
      self.beta = -0.145
      self.COEF1 = 2.0 * np.sqrt(2.) / 3.0
      self.COEF2 = -2.0 * np.sqrt(2.) / 3.0
      self.COEF3 = 2.0 * np.sqrt(2.) / 3.0
      self.SIGN = -taucharge
      self.doSystematic = False
      self.systType = "UP"
      
      self.debug = False

      RotVector = TauA1andProd[0].Vect()
      
      ReferenceFrame.SetVect(self.Rotate(ReferenceFrame.Vect(), RotVector))

      self._tauAlongZLabFrame = TauA1andProd[0]
      self._tauAlongZLabFrame.SetVect(self.Rotate(self._tauAlongZLabFrame.Vect(), RotVector))   

      TauA1andProd_RF = []
      for i in range(len(TauA1andProd)):
         Rotated = TauA1andProd[i]
         Rotated.SetVect(self.Rotate(Rotated.Vect(), RotVector))
         TauA1andProd_RF.append(Rotated)

      self.LFosPionLV = TauA1andProd[1]
      self.LFss1pionLV = TauA1andProd[2]
      self.LFss2pionLV = TauA1andProd[3]
      self.LFa1LV = self.LFosPionLV + self.LFss1pionLV + self.LFss2pionLV
      self.LFtauLV = TauA1andProd[0]
      self.LFQ = self.LFa1LV.M()

      self._osPionLV = TauA1andProd_RF[1]
      self._ss1pionLV = TauA1andProd_RF[2]
      self._ss2pionLV = TauA1andProd_RF[3]
      self._a1LV = self._osPionLV + self._ss1pionLV + self._ss2pionLV
      self._tauLV = TauA1andProd_RF[0]
      self._nuLV = self._tauLV - self._a1LV
      self._s12 = self._ss1pionLV + self._ss2pionLV
      self._s13 = self._ss1pionLV + self._osPionLV
      self._s23 = self._ss2pionLV + self._osPionLV
      self._s1 = self._s23.M2()
      self._s2 = self._s13.M2()
      self._s3 = self._s12.M2()
      self._Q = self._a1LV.M()

   def Rotate(self,Lvec, Rot):

      vec = Lvec
      vec.RotateZ(0.5*math.pi - Rot.Phi())
      vec.RotateX(Rot.Theta())
      
      return vec 

   def PVC(self):
      q1 = self._ss1pionLV
      q2 = self._ss2pionLV
      q3 = self._osPionLV
      
      a1 = q1 + q2 + q3
      N = self._nuLV
      P = self._tauLV
      s1 = (q2 + q3).M2()
      s2 = (q1 + q3).M2()
      s3 = (q1 + q2).M2()
   
      vec1 = q2 - q3 - a1 * (a1 * (q2 - q3) / a1.M2())
      vec2 = q3 - q1 - a1 * (a1 * (q3 - q1) / a1.M2())
      vec3 = q1 - q2 - a1 * (a1 * (q1 - q2) / a1.M2())
      
      F1 = ROOT.TComplex(self.COEF1) * self.F3PI(1, a1.M2(), s1, s2)
      F2 = ROOT.TComplex(self.COEF2) * self.F3PI(2, a1.M2(), s2, s1)
      F3 = ROOT.TComplex(self.COEF3) * self.F3PI(3, a1.M2(), s3, s1)
   
      HADCUR = [
           ROOT.TComplex(vec1.E()) * F1 + ROOT.TComplex(vec2.E()) * F2 + ROOT.TComplex(vec3.E()) * F3,  # energy component goes first
           ROOT.TComplex(vec1.Px()) * F1 + ROOT.TComplex(vec2.Px()) * F2 + ROOT.TComplex(vec3.Px()) * F3,
           ROOT.TComplex(vec1.Py()) * F1 + ROOT.TComplex(vec2.Py()) * F2 + ROOT.TComplex(vec3.Py()) * F3,
           ROOT.TComplex(vec1.Pz()) * F1 + ROOT.TComplex(vec2.Pz()) * F2 + ROOT.TComplex(vec3.Pz()) * F3
      ]
   
      HADCURC = [
           self.Conjugate(ROOT.TComplex(vec1.E()) * F1 + ROOT.TComplex(vec2.E()) * F2 + ROOT.TComplex(vec3.E()) * F3),  # energy component goes first
           self.Conjugate(ROOT.TComplex(vec1.Px()) * F1 + ROOT.TComplex(vec2.Px()) * F2 + ROOT.TComplex(vec3.Px()) * F3),
           self.Conjugate(ROOT.TComplex(vec1.Py()) * F1 + ROOT.TComplex(vec2.Py()) * F2 + ROOT.TComplex(vec3.Py()) * F3),
           self.Conjugate(ROOT.TComplex(vec1.Pz()) * F1 + ROOT.TComplex(vec2.Pz()) * F2 + ROOT.TComplex(vec3.Pz()) * F3)
      ]
      
      CLV = self.CLVEC(HADCUR, HADCURC, N)
      CLA = self.CLAXI(HADCUR, HADCURC, N)
      
      omega = P * CLV - P * CLA
      return (P.M() * P.M() * (CLA - CLV) - P * (P * CLA - P * CLV)) * (1 / omega / P.M())


   def F3PI(self, IFORM, QQ, SA, SB):
      MRO = 0.7743;
      GRO = 0.1491;
      MRP = 1.370 ;
      GRP = 0.386 ;
      MF2 = 1.275;
      GF2 = 0.185;
      MF0 = 1.186;
      GF0 = 0.350;
      MSG = 0.860;
      GSG = 0.880;
      MPIZ = self.mpi0;
      MPIC = self.mpi;   

      IDK = 1
      if IDK == 1:
         M1 = MPIZ
         M2 = MPIZ
         M3 = MPIC 
      if IDK == 2:
         M1 = MPIC
         M2 = MPIC
         M3 = MPIC     

      M1SQ = M1 * M1
      M2SQ = M2 * M2 
      M3SQ = M3 * M3   

      db2 = 0.094    
      db3 = 0.094  
      db4 = 0.296  
      db5 = 0.167  
      db6 = 0.284  
      db7 = 0.148     

      dph2 = 0.253  
      dph3 = 0.104  
      dph4 = 0.170  
      dph5 = 0.104  
      dph6 = 0.036  
      dph7 = 0.063        

      scale = 0
      if self.doSystematic:
         if self.systType == "UP":
            scale = 1
         if self.systType == "DOWN":
            scale = -1

      BT1 = ROOT.TComplex(1.0, 0.0)
      BT2 = ROOT.TComplex(0.12 + scale * db2, 0.0) * ROOT.TComplex(1, (0.99 + scale * dph2) * math.pi, True)
      BT3 = ROOT.TComplex(0.37 + scale * db3, 0.0) * ROOT.TComplex(1, (-0.15 + scale * dph3) * math.pi, True)
      BT4 = ROOT.TComplex(0.87 + scale * db4, 0.0) * ROOT.TComplex(1, (0.53 + scale * dph4) * math.pi, True)
      BT5 = ROOT.TComplex(0.71 + scale * db5, 0.0) * ROOT.TComplex(1, (0.56 + scale * dph5) * math.pi, True)
      BT6 = ROOT.TComplex(2.10 + scale * db6, 0.0) * ROOT.TComplex(1, (0.23 + scale * dph6) * math.pi, True)
      BT7 = ROOT.TComplex(0.77 + scale * db7, 0.0) * ROOT.TComplex(1, (-0.54 + scale * dph7) * math.pi, True)

      F3PIFactor = ROOT.TComplex(0.0, 0.0)     

      if IDK == 2:
         if(IFORM == 1 or IFORM == 2):
            S1 = SA
            S2 = SB
            S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ    
            #Lorentz invariants for all the contributions:
            F134 = -(1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F15A = -(1./2.)*((S2-M2SQ)-(S3-M3SQ))
            F15B = -(1./18.)*(QQ-M2SQ+S2)*(2.*M1SQ+2.*M3SQ-S2)/S2
            F167 = -(2./3.)  
            
            #Breit Wigners for all the contributions:  
            FRO1 = self.BWIGML(S1,MRO,GRO,M2,M3,1)
            FRP1 = self.BWIGML(S1,MRP,GRP,M2,M3,1)
            FRO2 = self.BWIGML(S2,MRO,GRO,M3,M1,1)
            FRP2 = self.BWIGML(S2,MRP,GRP,M3,M1,1)
            FF21 = self.BWIGML(S1,MF2,GF2,M2,M3,2)
            FF22 = self.BWIGML(S2,MF2,GF2,M3,M1,2)
            FSG2 = self.BWIGML(S2,MSG,GSG,M3,M1,0)
            FF02 = self.BWIGML(S2,MF0,GF0,M3,M1,0)

            F3PIFactor = BT1*FRO1 + BT2*FRP1 + BT3*ROOT.TComplex(F134,0.)*FRO2 + BT4*ROOT.TComplex(F134,0.)*FRP2 - BT5*ROOT.TComplex(F15A,0.)*FF21 - BT5*ROOT.TComplex(F15B,0.)*FF22 - BT6*ROOT.TComplex(F167,0.)*FSG2 - BT7*ROOT.TComplex(F167,0.)*FF02            

         elif (IFORM == 3):
            S3 = SA
            S1 = SB
            S2 = QQ-SA-SB+M1SQ+M2SQ+M3SQ
            #Lorentz invariants for all the contributions:
            F34A = (1./3.)*((S2-M2SQ)-(S3-M3SQ))
            F34B = (1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F35A = -(1./18.)*(QQ-M1SQ+S1)*(2.*M2SQ+2.*M3SQ-S1)/S1
            F35B =  (1./18.)*(QQ-M2SQ+S2)*(2.*M3SQ+2.*M1SQ-S2)/S2
            F36A = -(2./3.)
            F36B =  (2./3.)
            
            FRO1 = self.BWIGML(S1,MRO,GRO,M2,M3,1)
            FRP1 = self.BWIGML(S1,MRP,GRP,M2,M3,1)
            FRO2 = self.BWIGML(S2,MRO,GRO,M3,M1,1)
            FRP2 = self.BWIGML(S2,MRP,GRP,M3,M1,1)
            FF21 = self.BWIGML(S1,MF2,GF2,M2,M3,2)
            FF22 = self.BWIGML(S2,MF2,GF2,M3,M1,2)
            FSG1 = self.BWIGML(S1,MSG,GSG,M2,M3,0)
            FSG2 = self.BWIGML(S2,MSG,GSG,M3,M1,0)
            FF01 = self.BWIGML(S1,MF0,GF0,M2,M3,0)
            FF02 = self.BWIGML(S2,MF0,GF0,M3,M1,0)
            
            F3PIFactor = BT3*(ROOT.TComplex(F34A,0.)*FRO1+ROOT.TComplex(F34B,0.)*FRO2)+BT4*(ROOT.TComplex(F34A,0.)*FRP1+ROOT.TComplex(F34B,0.)*FRP2)-BT5*(ROOT.TComplex(F35A,0.)*FF21+ROOT.TComplex(F35B,0.)*FF22)-BT6*(ROOT.TComplex(F36A,0.)*FSG1+ROOT.TComplex(F36B,0.)*FSG2)-BT7*(ROOT.TComplex(F36A,0.)*FF01+ROOT.TComplex(F36B,0.)*FF02)
      if IDK == 1:  
         if(IFORM == 1 or IFORM == 2):
            S1 = SA
            S2 = SB
            S3 = QQ-SA-SB+M1SQ+M2SQ+M3SQ   

            F134 = -(1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F150 =  (1./18.)*(QQ-M3SQ+S3)*(2.*M1SQ+2.*M2SQ-S3)/S3
            F167 =  (2./3.)    

            FRO1 = self.BWIGML(S1,MRO,GRO,M2,M3,1)
            FRP1 = self.BWIGML(S1,MRP,GRP,M2,M3,1)
            FRO2 = self.BWIGML(S2,MRO,GRO,M3,M1,1)
            FRP2 = self.BWIGML(S2,MRP,GRP,M3,M1,1)
            FF23 = self.BWIGML(S3,MF2,GF2,M1,M2,2)
            FSG3 = self.BWIGML(S3,MSG,GSG,M1,M2,0)
            FF03 = self.BWIGML(S3,MF0,GF0,M1,M2,0)

            F3PIFactor = BT1*FRO1+BT2*FRP1+BT3*ROOT.TComplex(F134,0.)*FRO2+BT4*ROOT.TComplex(F134,0.)*FRP2+BT5*ROOT.TComplex(F150,0.)*FF23+BT6*ROOT.TComplex(F167,0.)*FSG3+BT7*ROOT.TComplex(F167,0.)*FF03            
         
         if(IFORM == 1 or IFORM == 2):
            S3 = SA
            S1 = SB
            S2 = QQ-SA-SB+M1SQ+M2SQ+M3SQ
            F34A = (1./3.)*((S2-M2SQ)-(S3-M3SQ))
            F34B = (1./3.)*((S3-M3SQ)-(S1-M1SQ))
            F35  =-(1./2.)*((S1-M1SQ)-(S2-M2SQ))           

            FRO1 = self.BWIGML(S1,MRO,GRO,M2,M3,1)
            FRP1 = self.BWIGML(S1,MRP,GRP,M2,M3,1)
            FRO2 = self.BWIGML(S2,MRO,GRO,M3,M1,1)
            FRP2 = self.BWIGML(S2,MRP,GRP,M3,M1,1)
            FF23 = self.BWIGML(S3,MF2,GF2,M1,M2,2)        
         
            F3PIFactor = BT3*(ROOT.TComplex(F34A,0.)*FRO1+ROOT.TComplex(F34B,0.)*FRO2)+BT4*(ROOT.TComplex(F34A,0.)*FRP1+ROOT.TComplex(F34B,0.)*FRP2)+BT5*ROOT.TComplex(F35,0.)*FF23
            
      FORMA1 = self.FA1A1P(QQ)            
      return F3PIFactor*FORMA1
      
           
   def BWIGML(self, S, M, G, m1, m2, L):
      MP = (m1+m2)**2
      MM = (m1-m2)**2
      MSQ = M*M;
      W = math.sqrt(S)
      WGS =0.0
      if(W > m1 + m2):
         QS = math.sqrt(abs((S - MP)*(S-MM)))/W
         QM = math.sqrt(abs((MSQ - MP)*(MSQ - MM)))/W
         IPOW = 2*L + 1
         WGS = G*(MSQ/W)*(QS/QM)**(IPOW)
         
      out = ROOT.TComplex(MSQ,0)/ROOT.TComplex(MSQ-S, -WGS)
      return out      
      
   def FA1A1P(self, XMSQ):
      XM1 = 1.275000
      XG1 =0.700
      XM2 = 1.461000
      XG2 = 0.250
      BET = ROOT.TComplex(0.,0.)
      
      GG1 = XM1*XG1/(1.3281*0.806)
      GG2 = XM2*XG2/(1.3281*0.806)
      XM1SQ = XM1*XM1
      XM2SQ = XM2*XM2
      
      GF = self.WGA1(XMSQ)
      FG1 = GG1*GF
      FG2 = GG2*GF
      F1 = ROOT.TComplex(-XM1SQ,0.0)/ROOT.TComplex(XMSQ-XM1SQ,FG1)
      F2 = ROOT.TComplex(-XM2SQ,0.0)/ROOT.TComplex(XMSQ-XM2SQ,FG2)
      FA1A1P = F1+BET*F2
      
      return FA1A1P

   def WGA1(self, QQ):
      MKST = 0.894
      MK = 0.496
      MK1SQ = (MKST+MK)*(MKST+MK)
      MK2SQ = (MKST-MK)*(MKST-MK)
      
      C3PI = 0.2384*0.2384
      CKST = 4.7621*4.7621*C3PI
      
      S = QQ
      WG3PIC = self.WGA1C(S)
      WG3PIN = self.WGA1N(S)

      GKST = 0.0
      if (S > MK1SQ): GKST = math.sqrt((S-MK1SQ)*(S-MK2SQ))/(2.*S)
      
      return C3PI*(WG3PIC+WG3PIN)+CKST*GKST

   def WGA1(self, S):
      Q0 = 5.80900
      Q1 = -3.00980
      Q2 = 4.57920
      P0 = -13.91400
      P1 = 27.67900
      P2 = -13.39300
      P3 = 3.19240
      P4 = -0.10487
      STH= 0.1753
      
      if (S > STH):
         G1_IM = 0.0
      elif (S > STH and S < 0.823):
         G1_IM = Q0 * (S - STH)**3 * (1. + Q1 * (S - STH) + Q2 * (S - STH)**2)
      else:
         G1_IM = P0 + P1*S + P2*S*S+ P3*S*S*S + P4*S*S*S*S
         
      return G1_IM

   def WGA1N(self, S):
      Q0 = 6.28450
      Q1 = -2.95950
      Q2 = 4.33550
      P0 = -15.41100
      P1 = 32.08800
      P2 = -17.66600
      P3 = 4.93550
      P4 = -0.37498
      STH= 0.1676
      
      if (S > STH):
         G1_IM = 0.0
      elif (S > STH and S < 0.823):
         G1_IM = Q0 * (S - STH)**3 * (1. + Q1 * (S - STH) + Q2 * (S - STH)**2)
      else:
         G1_IM = P0 + P1*S + P2*S*S+ P3*S*S*S + P4*S*S*S*S
         
      return G1_IM      

   def Conjugate(self, a):
      return ROOT.TComplex(a.Re(), -a.Im())
      
   def CLVEC(self, H, HC, N):
      HN = H[0]*N.E() - H[1]*N.Px() - H[2]*N.Py() - H[3]*N.Pz()   
      HCN = HC[0]*N.E() - HC[1]*N.Px() - HC[2]*N.Py() - HC[3]*N.Pz()
      HH = (H[0]*HC[0] - H[1]*HC[1] - H[2]*HC[2] - H[3]*HC[3]).Re()
      
      PIVEC0 = 2*(2*(HN*HC[0]).Re()- HH*N.E())
      PIVEC1 = 2*(2*(HN*HC[1]).Re()- HH*N.Px())
      PIVEC2 = 2*(2*(HN*HC[2]).Re()- HH*N.Py())
      PIVEC3 = 2*(2*(HN*HC[3]).Re()- HH*N.Pz())
      
      return ROOT.TLorentzVector(PIVEC1, PIVEC2, PIVEC3, PIVEC0)
      
   def CLAXI(self, H, HC, N):
      a4 = HC[0]
      a1 = HC[1]
      a2 = HC[2]
      a3 = HC[3]

      b4 = H[0]
      b1 = H[1]
      b2 = H[2]
      b3 = H[3] 

      c4 = N.E()
      c1 = N.Px()
      c2 = N.Py()
      c3 = N.Pz()
      
      d34 = (a3*b4 - a4*b3).Im()
      d24 = (a2*b4 - a4*b2).Im() 
      d23 = (a2*b3 - a3*b2).Im()
      d14 = (a1*b4 - a4*b1).Im()
      d13 = (a1*b3 - a3*b1).Im()
      d12 = (a1*b2 - a2*b1).Im()

      PIAX0 = -self.SIGN*2*(-c1*d23 + c2*d13 - c3*d12)
      PIAX1 = self.SIGN*2*(c2*d34 - c3*d24 + c4*d23)
      PIAX2 = self.SIGN*2*(-c1*d34 + c3*d14 - c4*d13)
      PIAX3 = self.SIGN*2*(c1*d24 - c2*d14 + c4*d12)    

      return ROOT.TLorentzVector(PIAX1,PIAX2,PIAX3,PIAX0)      

   def BreitWigner(self, Q, Type):
      QQ=Q*Q
      m = self.Mass(Type)
      g  = self.Widths(Q,Type)
      re = (m*m*(m*m - QQ))/(((m*m - QQ)**2) + m*m*g*g)
      im = Q*g/(((m*m - QQ)**2) + m*m*g*g)
      
      return ROOT.TComplex(re,im)

   def Mass(self, Type):
      m = self.mrho
      if Type == "rhoprime": return self.mrhoprime
      if Type == "a1": return self.ma1
      if Type == "piprime": return self.mpiprime
      return m

   def Widths(self, Q, Type):
      QQ = Q*Q
      Gamma = self.Gamma0rho*self.mrho*(self.ppi(QQ)/self.ppi(self.mrho*self.mrho))**3/math.sqrt(QQ)
      if Type == "rhoprime": Gamma=self.Gamma0rhoprime*QQ/self.mrhoprime/self.mrhoprime
      if Type == "a1": Gamma = self.Gamma0a1*self.ga1(Q)/self.ga1(self.ma1)
      if Type == "piprime": Gamma = self.Gamma0piprime*(math.sqrt(QQ)/self.mpiprime)**5*((1-self.mrho*self.mrho/QQ)/(1-self.mrho*self.mrho/self.mpiprime/self.mpiprime))**3
      return Gamma
      
   def ppi(self, QQ):
      if (QQ < 4*self.mpi*self.mpi): print("Warning! Can not compute ppi(Q); root square <0 ; return nan")
      return 0.5*math.sqrt(QQ - 4*self.mpi*self.mpi)

   def ga1(self, Q):
      QQ = Q*Q
      if (QQ > (self.mrho + self.mpi)**2): return QQ * (1.623 + 10.38/QQ - 9.32/QQ/QQ   + 0.65/QQ/QQ/QQ)
      return 4.1*(QQ - 9*self.mpi*self.mpi)**3*(1 - 3.3*(QQ - 9*self.mpi*self.mpi)  + 5.8*(QQ - 9*self.mpi*self.mpi)**2)

   def f3(self, Q):
      return (self.coscab*2*math.sqrt(2)/3/self.fpi)*self.BreitWigner(Q, "a1")
