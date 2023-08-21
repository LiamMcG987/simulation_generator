import os
from datetime import datetime


class UpdateLog:

    geometry = {
        2: 'Axisymmetric',
        3: '3D'
    }

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    def __init__(self, log_file=None, log_file_mapping_params=None):
        os.chdir('../inp_populater/mapping_data_logging')
        self.log_file = log_file if log_file is not None else 'logfile.txt'
        self.log_file_mapping_params = log_file_mapping_params if log_file_mapping_params is not None else \
            'logfile_mapping_params.txt '

    def update_log_check(self, file):
        with open(self.log_file, 'a') as log:
            log.write("""\n{}
    Data checked for file {}.\n"""
                      .format(UpdateLog.now, file))

    def update_log_map(self, map_from, mf_shape, map_to, mt_shape, parameters):
        with open(self.log_file, 'a') as log:
            log.write("""\n{}
    {} ({}) mapped to {} ({}).\n"""
                      .format(UpdateLog.now,
                              UpdateLog.geometry.get(mf_shape), map_from,
                              UpdateLog.geometry.get(mt_shape), map_to))

            self.update_log_mapping_params(map_from, map_to, parameters)

    def update_log_write(self, original_inp):
        with open(self.log_file, 'a') as log:
            log.write("""\n{}
    {} rewritten for map_to file.
    New .inp file found with '_new' extension.\n"""
                      .format(UpdateLog.now,
                              original_inp))

    def update_log_mapping_params(self, map_from, map_to, parameters):
        with open(self.log_file_mapping_params, 'a') as log_map:
            log_map.write("""\n{}
    Data mapped from {} to {}.
    Parameters used:
    {}\n"""
                          .format(UpdateLog.now,
                                  map_from, map_to,
                                  parameters))
