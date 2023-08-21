# simulation_generator

automatic population of simulation files for Abaqus CAE and 3DExperience
requires inputs of bottle mesh, thickness maps, and modulus maps if desired - .xlsx, .csv, .mat all supported

GUI created also to enable easy solution
provides visualisation of mapped bottle to verify mapping contours and positioning of rigid bodies before committing to population
relevant translations required for rigid bodies, with option to flip orientation of bottle

part_creation_inp provides option to add further rigid bodies into library
requires simulation file with part, which will then be automatically extracted when name of part is inserted into script
