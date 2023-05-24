########################### GATEWAY HEADER #############################
#                           
#     _|      _|  _|_|_|  _|_|_|      _|_|      _|_|_|    _|_|    
#     _|_|  _|_|    _|    _|    _|  _|    _|  _|        _|    _|  
#     _|  _|  _|    _|    _|    _|  _|_|_|_|  _|        _|    _|  
#     _|      _|    _|    _|    _|  _|    _|  _|        _|    _|  
#     _|      _|  _|_|_|  _|_|_|    _|    _|    _|_|_|    _|_|  
#
#                                                   Version 6.0
#
########################################################################
#
#           See the MIDACO user manual for detailed information
#
########################################################################
#
#    Author (C) :   Dr. Martin Schlueter
#                   Information Initiative Center,
#                   Division of Large Scale Computing Systems,
#                   Hokkaido University, JAPAN.
#
#    Email :        info@midaco-solver.com
#
#    URL :          http://www.midaco-solver.com
#       
########################################################################

import ctypes; 
from ctypes import *
import os
import os.path
import numpy as np

########################################################################

def run( problem, option, key ):

 problem_function = problem['@']

 o =problem['o'] 
 n =problem['n'] 
 ni=problem['ni']
 m =problem['m'] 
 me=problem['me']

 xl=problem['xl']
 xu=problem['xu']
 x =problem['x']

 maxeval=option['maxeval']
 maxtime=option['maxtime']

 printeval=option['printeval']
 save2file=option['save2file']

 param=[0.0]*13

 param[ 0] = option['param1']  
 param[ 1] = option['param2']  
 param[ 2] = option['param3']  
 param[ 3] = option['param4']  
 param[ 4] = option['param5']  
 param[ 5] = option['param6']  
 param[ 6] = option['param7']  
 param[ 7] = option['param8']  
 param[ 8] = option['param9']  
 param[ 9] = option['param10'] 
 param[10] = option['param11'] 
 param[11] = option['param12'] 
 param[12] = option['param13']

 PARALLEL = option['parallel']

 if PARALLEL <= 1 : p = 1
 if PARALLEL  > 1 : p = PARALLEL

 ########################################################################
 ########################################################################
 ########################################################################
 ####################### IMPORT MIDACO LIBRARY ##########################
 ########################################################################
 ########################################################################
 ########################################################################
 
 # Specify name of the MIDACO library depending on OS
 if (os.name == "posix"):  lib_name = "midacopy.so"  # Linux//Mac/Cygwin
 else:                     lib_name = "midacopy.dll" # Windows
 # Specify path were the MIDACO library is expected  
 lib_path=os.path.dirname(os.path.abspath(__file__))+os.path.sep+lib_name
 # Assign CLIB as name for MIDACO library
 CLIB = ctypes.CDLL(lib_path)
 ########################################################################
 ######################### CALL MIDACO SOLVER ###########################
 ########################################################################
 key_ = c_char_p(key)  
 # Create c-types arguments and initialize workspace and flags
 n_  = pointer(c_long(n));  n__   = c_long(n);  xl_    = (c_double * n)()
 ni_ = pointer(c_long(ni)); ni__  = c_long(ni); xu_    = (c_double * n)()
 m_  = pointer(c_long(m));  m__   = c_long(m);  x_     = (c_double * n)()
 me_ = pointer(c_long(me)); me___ = c_long(me); param_ = (c_double *13)()
 o_  = pointer(c_long(o));  o__   = c_long(o);        
 printeval_ = c_long(printeval);  iflag_ = pointer(c_long(0)) 
 save2file_ = c_long(save2file);  istop_ = pointer(c_long(0))
 maxeval_   = c_long(maxeval);        p__= c_long(p);
 maxtime_   = c_long(maxtime);        p_ = pointer(c_long(p))  
 lrw_ = pointer(c_long(120*n+20*m+20*o+20*p+p*(m+2*o)+o*o+5000));  
 rw_ =  (c_double * lrw_[0])(0.0)
 liw_ = pointer(c_long(3*n+p+1000));      
 iw_ =  (c_long *   liw_[0])(0)    
 if( o == 1 ): 
     lpf_ = pointer(c_long(1))
 if( o > 1 ): 
     lpf_ = pointer(c_long(1000*(o+m+n)+1+100))
     if( param[9] >= 1.0 ): 
       lpf_ = pointer(c_long(  int(param[9]) * (o+m+n) + 1))
     if( param[9] <=-1.0 ): 
       lpf_ = pointer(c_long( -int(param[9]) * (o+m+n) + 1))       
 pf_ =  (c_double * lpf_[0])(0.0)
############### for i in range(0, lpf_[0]):  pf_[i] = c_double(0.0)  
############### for i in range(0, lrw_[0]):  rw_[i] = c_double(0.0)    
############### for i in range(0, liw_[0]):  iw_[i] = c_long(0); 
 for i in range(0,13): param_[i] = c_double(param[i])
 for i in range(0, n): xl_[i]    = c_double(xl[i])  
 for i in range(0, n): xu_[i]    = c_double(xu[i])  
 for i in range(0, n): x_[i]     = c_double( x[i])
 f_ = (c_double * o)()
 if ( m > 0): g_ = (c_double * m)()
 if ( m ==0): g_ = (c_double * 1)() # Dummy for unconstrained problems
 ########################################################################





 if PARALLEL <= 1 :
   ########################################################################
   ########################################################################
   ########################################################################
   # Print MIDACO Head information
   CLIB.midaco_print(1,printeval_,save2file_,iflag_,istop_,f_,g_,x_,xl_,xu_,\
                     o__,n__,ni__,m__,me___,rw_,pf_,maxeval_,maxtime_,param_,p__,key_)      
   ########################################################################
   while True: # Call MIDACO by reverse communication loop 
   
   
     [ f_[:], g_[:] ] =  problem_function(x_[:]) # Evaluate F(X) and G(X)  


     # Check and repair NaN
     for i in range(0,o): 
          if np.isnan(f_[i]) : 
               f_[i] =  1.0e+33
     for i in range(0,m): 
          if np.isnan(g_[i]) : 
               g_[i] = -1.0e+33
     # Check and repair Inf
     for i in range(0,o): 
          if np.isinf(f_[i]) : 
               f_[i] =  1.0e+32
     for i in range(0,m): 
          if np.isinf(g_[i]) : 
               g_[i] = -1.0e+32  
                  

     CLIB.midaco(p_,o_,n_,ni_,m_,me_,x_,f_,g_,xl_,xu_, \
                 iflag_,istop_,param_,rw_,lrw_,iw_,liw_,pf_,lpf_,key_)
                 
     CLIB.midaco_print(2,printeval_,save2file_,iflag_,istop_, \
                       f_,g_,x_,xl_,xu_,o__,n__,ni__,m__,me___,rw_,pf_, \
                       maxeval_,maxtime_,param_,p__,key_)              
     
     if istop_[0] != 0: break          
   ########################################################################
   ########################################################################
   ########################################################################        






 if PARALLEL > 1 :
   ########################################################################
   ########################################################################
   ########################################################################   
   # Pre-Allocate A and B for speed
   A  = [[None]*n]*p
   B  = [[None],[None]*m]*p
   # Create paralle arrays for f,g and x
   po = p*o; fff_ = (c_double * po)()
   pm = p*m; ggg_ = (c_double * pm)()
   pn = p*n; xxx_ = (c_double * pn)()
   #
   #  Copy starting point X into XXX parallel array
   #
   for c in range(0,p): 
     if c <= 0 : # First solution entry for XXX array
       for i in range(0, n): xxx_[c*n+i] = c_double(x[i])
     else : # Second and all remaining solution entries for XXX
       for i in range(0, n): xxx_[c*n+i] = c_double(x[i])

       #
       # Special case: fill up XXX array with small perturbations of X
       #

       # import random
       # PERTURBATION = 0.001
       # for i in range(0, n): xxx_[c*n+i] = c_double(x[i]) 
       # for i in range(0, n-ni): xxx_[c*n+i] = xxx_[c*n+i] + random.random()*(xu[i]-xl[i])*PERTURBATION
       # for i in range(0, n-ni): xxx_[c*n+i] = xxx_[c*n+i] - random.random()*(xu[i]-xl[i])*PERTURBATION
       # for i in range(0, n-ni): 
       #  if xxx_[c*n+i] < xl[i] : xxx_[c*n+i] = xl[i] # repair lower bound violation
       # for i in range(0, n-ni): 
       #  if xxx_[c*n+i] > xu[i] : xxx_[c*n+i] = xu[i] # repair upper bound violation


   # Start the parallel pool
   from multiprocessing import Pool
   pool   = Pool(processes=p); 
   #print '\n *** Running MIDACO in Parallel Mode: P = ',p,' ***'
   ########################################################################
   ########################################################################
   ########################################################################
   # Print MIDACO Head information
   CLIB.midaco_print(1,printeval_,save2file_,iflag_,istop_,f_,g_,x_,xl_,xu_,\
                     o__,n__,ni__,m__,me___,rw_,pf_,maxeval_,maxtime_,param_,p__,key_)      
   ########################################################################
   while True: # Call MIDACO by reverse communication loop 
           
     # Store x in A   
     for c in range(0,p): 
       x = [None]*n
       for i in range(0,n): x[i] = xxx_[c*n+i]
       A[c] = x
       
     # Evaluate problem function in parallel  
     B = pool.map( problem_function, A)
         
     # Get f(x) and g(x) out of B
     for c in range(0,p): 
        
       for j in range(0,o): fff_[c*o+j] = c_double(B[c][0][j])
       for j in range(0,m): ggg_[c*m+j] = c_double(B[c][1][j])


       # Check and repair NaN
       for i in range(0,o*p): 
            if np.isnan(fff_[i]) : 
                 fff_[i] =  1.0e+33
       for i in range(0,m*p): 
            if np.isnan(ggg_[i]) : 
                 ggg_[i] = -1.0e+33
       # Check and repair Inf
       for i in range(0,o*p): 
            if np.isinf(fff_[i]) : 
                 fff_[i] =  1.0e+32
       for i in range(0,m*p): 
            if np.isinf(ggg_[i]) : 
                 ggg_[i] = -1.0e+32  

                      
     CLIB.midaco(p_,o_,n_,ni_,m_,me_,xxx_,fff_,ggg_,xl_,xu_, \
                 iflag_,istop_,param_,rw_,lrw_,iw_,liw_,pf_,lpf_,key_)     
                 
     CLIB.midaco_print(2,printeval_,save2file_,iflag_,istop_, \
                       fff_,ggg_,xxx_,xl_,xu_,o__,n__,ni__,m__,me___,rw_,pf_, \
                       maxeval_,maxtime_,param_,p__,key_)                                 
     
     if istop_[0] != 0: break  
   ########################################################################
   ########################################################################
   ########################################################################  
   # Get solution x out of xxx  
   for i in range(0,o): f_[i] = fff_[i]
   for i in range(0,m): g_[i] = ggg_[i]
   for i in range(0,n): x_[i] = xxx_[i]                          

   pool.close()
   pool.terminate() 




 ########################################################################
 #################### RETURN ARGUMENTS = SOLUTION #######################
 ########################################################################
 
 f = [0.0]*o 
 g = [0.0]*m 
 x = [0.0]*n 

 for i in range(0,o): f[i] = f_[i]
 for i in range(0,m): g[i] = g_[i]
 for i in range(0,n): x[i] = x_[i]     
    
 solution = {}

 solution['f'] = f
 solution['g'] = g
 solution['x'] = x
 solution['iflag'] = iflag_[0]

 return solution 
 ########################################################################
 ############################ END OF FILE ###############################
 #######################################################################
         