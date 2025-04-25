import os
import numpy as np
import pandas as pd
import subprocess
from utils import Rod, R2atom

MONOMER_LIST = ['C5','C6','C9','C28','C32','C33']
############################汎用関数###########################
def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A2,A3):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono=pd.read_csv(f'~/Working/FF_calc/polyacene/monomer/{monomer_name}_dreiding.csv')
    atoms_array_xyzR=df_mono[['x','y','z','atom']].values
    
    ex = np.array([1.,0.,0.]); ey = np.array([0.,1.,0.]); ez = np.array([0.,0.,1.])

    xyz_array = atoms_array_xyzR[:,:3]
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)
    xyz_array = xyz_array + T_vec
    R_array = atoms_array_xyzR[:,3].reshape((-1,1))
    
    return np.concatenate([xyz_array,R_array],axis=1)
        
def get_xyzR_lines(xyza_array,file_description,machine_type):
    if machine_type==1:
        mp_num = 40
    elif machine_type==2:
        mp_num = 52
    lines = [     
        '%mem=15GB\n',
        f'%nproc={mp_num}\n',
        '# dreiding=QEq geom=connectivity\n',###汎関数や基底関数系は適宜変更する
        '\n',
        file_description+'\n',
        '\n',
        '0 1\n'
    ]
    for x,y,z,atom in xyza_array:
        line = '{} {} {} {}\n'.format(atom,x,y,z)     
        lines.append(line)
    
    lines.append('\n')
    lines_connectivity=['1 2 1.5 3 1.5 7 1.5 \n', '2 4 1.5 9 1.5 \n', '3 5 1.5 11 1.0 \n', '4 6 1.5 14 1.0 \n', '5 6 1.5 16 1.0 \n', '6 15 1.0 \n', '7 8 1.5 12 1.0 \n', '8 10 1.5 17 1.5 \n', '9 10 1.5 13 1.0 \n', '10 19 1.5 \n', '11 \n', '12 \n', '13 \n', '14 \n', '15 \n', '16 \n', '17 18 1.5 21 1.0 \n', '18 20 1.5 23 1.5 \n',
                        '19 20 1.5 22 1.0 \n', '20 25 1.5 \n', '21 \n', '22 \n', '23 24 1.5 27 1.0 \n', '24 26 1.5 29 1.5 \n', '25 26 1.5 28 1.0 \n', '26 31 1.5 \n', '27 \n', '28 \n', '29 30 1.5 33 1.0 \n', '30 32 1.5 34 1.0 \n', '31 32 1.5 36 1.0 \n', '32 35 1.0 \n', '33 \n', '34 \n', '35 \n', '36 \n',
                        '37 38 1.5 39 1.5 43 1.5 \n', '38 40 1.5 45 1.5 \n', '39 41 1.5 47 1.0 \n', '40 42 1.5 50 1.0 \n', '41 42 1.5 52 1.0 \n', '42 51 1.0 \n', '43 44 1.5 48 1.0 \n', '44 46 1.5 53 1.5 \n', '45 46 1.5 49 1.0 \n', '46 55 1.5 \n', '47 \n', '48 \n', '49 \n', '50 \n', '51 \n', '52 \n', '53 54 1.5 57 1.0 \n', '54 56 1.5 59 1.5 \n',
                        '55 56 1.5 58 1.0 \n', '56 61 1.5 \n', '57 \n', '58 \n', '59 60 1.5 63 1.0 \n', '60 62 1.5 65 1.5 \n', '61 62 1.5 64 1.0 \n', '62 67 1.5 \n', '63 \n', '64 \n', '65 66 1.5 69 1.0 \n', '66 68 1.5 70 1.0 \n', '67 68 1.5 72 1.0 \n', '68 71 1.0 \n', '69 \n', '70 \n', '71 \n', '72 \n']
    for line in lines_connectivity:
        lines.append(line)
    return lines

# 実行ファイル作成
def get_one_exe(file_name,machine_type):
    file_basename = os.path.splitext(file_name)[0]
    #mkdir
    if machine_type==1:
        gr_num = 1; mp_num = 40
    elif machine_type==2:
        gr_num = 2; mp_num = 52
    cc_list=[
        '#!/bin/sh \n',
        '#$ -S /bin/sh \n',
        '#$ -cwd \n',
        '#$ -V \n',
        '#$ -q gr{}.q \n'.format(gr_num),
        '#$ -pe OpenMP {} \n'.format(mp_num),
        '\n',
        'hostname \n',
        '\n',
        'export g16root=/home/g03 \n',
        'source $g16root/g16/bsd/g16.profile \n',
        '\n',
        'export GAUSS_SCRDIR=/scr/$JOB_ID \n',
        'mkdir /scr/$JOB_ID \n',
        '\n',
        'g16 < {}.inp > {}.log \n'.format(file_basename,file_basename),
        '\n',
        'rm -rf /scr/$JOB_ID \n',
        '\n',
        '\n',
        '#sleep 5 \n'
#          '#sleep 500 \n'
            ]

    return cc_list

######################################## 特化関数 ########################################

##################gaussview##################
def make_xyzfile(monomer_name,params_dict,structure_type):
    a = params_dict.get('a',0.0)
    b = params_dict.get('b',0.0); z = params_dict.get('z',0.0)
    A2 = params_dict.get('A2',0.0); A3 = params_dict.get('theta',0.0)

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3)
    
    monomer_array_p1 = get_monomer_xyzR(monomer_name,a,0,0,A2,A3)##1,2がb方向
    monomer_array_p2 = get_monomer_xyzR(monomer_name,0,b,2*z,A2,A3)##1,2がb方向
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a/2,b/2,z,A2,-A3)##1,2がb方向
    monomer_array_t2 = get_monomer_xyzR(monomer_name,-a/2,b/2,z,A2,-A3)##1,2がb方向
    
    xyz_list=['400 \n','polyacene9 \n']##4分子のxyzファイルを作成
    
    if structure_type == 1:##隣接8分子について対称性より3分子でエネルギー計算
        monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p1],axis=0)
    elif structure_type == 2:##隣接8分子について対称性より3分子でエネルギー計算
        monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p2],axis=0)
    elif structure_type == 3:##隣接8分子について対称性より3分子でエネルギー計算
        monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p1,monomer_array_p2,monomer_array_t1,monomer_array_t2],axis=0)
    
    for x,y,z,R in monomers_array_4:
        atom = R2atom(R)
        line = '{} {} {} {}\n'.format(atom,x,y,z)     
        xyz_list.append(line)
    
    return xyz_list

def make_xyz(monomer_name,params_dict,structure_type):
    xyzfile_name = ''
    xyzfile_name += monomer_name
    for key,val in params_dict.items():
        if key in ['a','b','z']:
            val = np.round(val,2)
        elif key in ['A1','A2','theta']:
            val = int(val)
        xyzfile_name += '_{}={}'.format(key,val)
    return xyzfile_name + f'_{structure_type}.xyz'

def make_gjf_xyz(auto_dir,monomer_name,params_dict,machine_type,structure_type):
    a = params_dict.get('a',0.0); b = params_dict.get('b',0.0); z = params_dict.get('z',0.0)
    A2 = params_dict.get('A2',0.0); A3 = params_dict.get('theta',0.0)

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3)
    monomer_array_p1 = get_monomer_xyzR(monomer_name,a,0,0,A2,A3)##1,2がb方向
    monomer_array_p2 = get_monomer_xyzR(monomer_name,0,b,2*z,A2,A3)##1,2がb方向
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a/2,b/2,z,A2,-A3)##1,2がb方向
    monomer_array_t2 = get_monomer_xyzR(monomer_name,-a/2,b/2,z,A2,-A3)##1,2がb方向
    
    dimer_array_p1 = np.concatenate([monomer_array_i,monomer_array_p1]);dimer_array_p2 = np.concatenate([monomer_array_i,monomer_array_p2])
    dimer_array_t1 = np.concatenate([monomer_array_i,monomer_array_t1]);dimer_array_t2 = np.concatenate([monomer_array_i,monomer_array_t2])
    
    file_description = f'{monomer_name}_theta={A3}_a={a}_b={b}_z={z}'
    line_list_dimer_p1 = get_xyzR_lines(dimer_array_p1,file_description+'_p1',machine_type);line_list_dimer_p2 = get_xyzR_lines(dimer_array_p2,file_description+'_p2',machine_type)
    line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1,file_description+'_t1',machine_type);line_list_dimer_t2 = get_xyzR_lines(dimer_array_t2,file_description+'_t2',machine_type)
    
    if structure_type == 1:##隣接8分子について対称性より3分子でエネルギー計算
        gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_p1 + ['\n\n\n']
    elif structure_type == 2:##隣接8分子について対称性より3分子でエネルギー計算
        gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_p2 + ['\n\n\n']
    elif structure_type == 3:##隣接8分子について対称性より3分子でエネルギー計算
        gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_t1 + ['\n\n\n']
    
    file_name = get_file_name_from_dict(monomer_name,params_dict,structure_type)
    os.makedirs(os.path.join(auto_dir,'gaussian'),exist_ok=True)
    gij_xyz_path = os.path.join(auto_dir,'gaussian',file_name)
    with open(gij_xyz_path,'w') as f:
        f.writelines(gij_xyz_lines)
    
    return file_name

def get_file_name_from_dict(monomer_name,params_dict,structure_type):
    file_name = ''
    file_name += monomer_name
    for key,val in params_dict.items():
        if key in ['a','b','z']:
            val = val
        elif key in ['A2','theta']:
            val = int(val)
        file_name += '_{}={}'.format(key,val)
    return file_name + f'_{structure_type}.inp'
    
def exec_gjf(auto_dir, monomer_name, params_dict, machine_type,structure_type,isTest=True):
    inp_dir = os.path.join(auto_dir,'gaussian')
    xyz_dir = os.path.join(auto_dir,'gaussview')
    print(params_dict)
    
    xyzfile_name = make_xyz(monomer_name, params_dict,structure_type)
    xyz_path = os.path.join(xyz_dir,xyzfile_name)
    xyz_list = make_xyzfile(monomer_name,params_dict,structure_type)
    with open(xyz_path,'w') as f:
        f.writelines(xyz_list)
    
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict,machine_type,structure_type)
    cc_list = get_one_exe(file_name,machine_type)
    sh_filename = os.path.splitext(file_name)[0]+'.r1'
    sh_path = os.path.join(inp_dir,sh_filename)
    with open(sh_path,'w') as f:
        f.writelines(cc_list)
    if not(isTest):
        subprocess.run(['qsub',sh_path])
    log_file_name = os.path.splitext(file_name)[0]+'.log'
    return log_file_name
    
############################################################################################