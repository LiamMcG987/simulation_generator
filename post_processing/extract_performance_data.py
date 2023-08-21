
from odbAccess import *
from abaqusConstants import *
import csv
import os, glob
import numpy as np
import pdb


# Methods

def get_sim_info(odb):
    """ Specify inputs for top load calculations. """

    instance = odb.rootAssembly.instances.keys()[0]
    instance_nodeSets = odb.rootAssembly.instances[instance].nodeSets.keys()
    instance_nodeSets = [s.lower() for s in instance_nodeSets]
    combined_nodeSets = '\t'.join(instance_nodeSets)
    if 'top_load' in combined_nodeSets:
        sim_type = 'top_load'
        axis = 2
        idx = 0
        key_str = 'REFERENCE_POINT'
        print 'Simulation: {}, type: Top load'.format(file)
    elif 'side_load' in combined_nodeSets:
        sim_type = 'side_load'
        axis = 0
        idx = 1
        key_str = 'REFERENCE_POINT'
        print 'Simulation: {}, type: Side load'.format(file)
    elif 'squeeze' in combined_nodeSets:
        sim_type = 'squeeze'
        axis = 1
        idx = 1
        key_str = 'REFERENCE_POINT'
        print 'Simulation: {}, type: Squeeze'.format(file)
    elif 'bulge' in combined_nodeSets:
        sim_type = 'bulge'
        axis = 1
        idx = 0
        key_str = 'ALL NODES'
        print 'Simulation: {}, type: Bulge'.format(file)
    return sim_type, axis, idx, key_str


# Main

def extract_data(file):
    """ Read odb file and extract key information. """

    # Open odb file and specify key information
    file_name, file_ext = os.path.splitext(file)
    odb = openOdb(file)
    stepname = odb.steps.keys()[0]
    step = odb.steps[stepname]
    frames = step.frames
    # Get simulation information (type, axis and key)
    sim_type, axis, idx, key_str = get_sim_info(odb)
    # Specify nodes of interest
    nodeSets = odb.rootAssembly.nodeSets.keys()
    node_str = [s for s in nodeSets if key_str in s][idx]
    nodeSet = odb.rootAssembly.nodeSets[node_str]
    if not sim_type == 'bulge':
        # Extract force, displacement data over all frames
        disp_force = np.zeros((len(frames), 2))
        for i in range(len(step.frames)):
            disp_force[i, 0] = abs(frames[i].fieldOutputs['U'].getSubset(region=nodeSet).values[0].data[axis])
            disp_force[i, 1] = abs(frames[i].fieldOutputs['RF'].getSubset(region=nodeSet).values[0].data[axis])
            print 'Iteration ' + str(i) + '/' + str(len(step.frames))
        np.savetxt('{}_disp_force.csv'.format(file_name), disp_force, delimiter=',')
    else:
        # Extract maximum displacement
        coords = frames[-1].fieldOutputs['U'].getSubset(region=nodeSet).values
        all_coords = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            all_coords[i] = coord.data[axis]
        max_deflection = np.max(np.abs(all_coords))
        with open('{}_disp_force.csv'.format(file_name), 'w') as f:
            writer = csv.writer(f)
            writer.writerow([max_deflection])


if __name__ == "__main__":
    # Run script for all files in directory
    files = glob.glob("*.odb")
    for file in files:
        file_name, file_ext = os.path.splitext(file)
        if not os.path.exists('{}_disp_force.csv'.format(file_name)):
            print 'Extracting data for {}'.format(file_name)
            extract_data(file)
        else:
            print 'Data already extracted for {}'.format(file_name)
else:
    # Run script for single file
    file = sys.argv[-1]
    extract_data(file)
