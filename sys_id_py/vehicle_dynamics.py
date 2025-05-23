import math

def vehicle_dynamics_st(x, uInit, p, type):
    """
    vehicleDynamics_st - single-track vehicle dynamics
    reference point: center of mass

    Syntax:
        f = vehicleDynamics_st(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param uInit: vehicle input vector
        :param p: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations
    """
    
    # set gravity constant
    g = 9.81  #[m/s^2]

    #create equivalent bicycle parameters
    if type == "pacejka":
        B_f = p.C_Pf[0]
        C_f = p.C_Pf[1]
        D_f = p.C_Pf[2]
        E_f = p.C_Pf[3]
        B_r = p.C_Pr[0]
        C_r = p.C_Pr[1]
        D_r = p.C_Pr[2]
        E_r = p.C_Pr[3]
    elif type == "linear":
        C_Sf = p.C_Sf #-p.tire.p_ky1/p.tire.p_dy1  
        C_Sr = p.C_Sr #-p.tire.p_ky1/p.tire.p_dy1  
    lf = p.l_f
    lr = p.l_r
    h = p.h_cg 
    m = p.m 
    I = p.I_z

    #states
    #x0 = x-position in a global coordinate system
    #x1 = y-position in a global coordinate system
    #x2 = yaw angle
    #x3 = velocity in x-direction
    #x4 = velocity in y direction
    #x5 = yaw rate

    #u1 = steering angle
    #u2 = longitudinal acceleration

    u = uInit

    # system dynamics

    # compute lateral tire slip angles
    alpha_f = -math.atan((x[4] + x[5] * lf) / x[3]) + u[0] 
    alpha_r = -math.atan((x[4] - x[5] * lr) / x[3])


    # compute vertical tire forces
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)

    F_yf = F_yr = 0

    # combined slip lateral forces
    if type == "pacejka":
        F_yf = F_zf * D_f * math.sin(C_f * math.atan(B_f * alpha_f - E_f*(B_f * alpha_f - math.atan(B_f * alpha_f))))
        F_yr = F_zr * D_r * math.sin(C_r * math.atan(B_r * alpha_r - E_r*(B_r * alpha_r - math.atan(B_r * alpha_r))))
    elif type == "linear":
        F_yf = F_zf * C_Sf * alpha_f
        F_yr = F_zr * C_Sr * alpha_r

    f = [x[3]*math.cos(x[2]) - x[4]*math.sin(x[2]), 
        x[3]*math.sin(x[2]) + x[4]*math.cos(x[2]), 
        x[5], 
        u[1], 
        1/m * (F_yr + F_yf) - x[3] * x[5],
        1/I * (-lr * F_yr + lf * F_yf)] 
    return f

    #------------- END OF CODE --------------
