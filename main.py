from inp_populater.get_simulation_components import PopulateSimulation

if __name__ == '__main__':
    sbm_data = 'mowgli_750_34g.xlsx'
    perf_mesh = 'Reckitt_Mowgli_750ml_RIBS_34g_Bulge'
    simulation = 'side load'
    bottle_mass = 34
    translation_matrix = [[-54.15, 0, -116.74],  # 0, -26.9, -134.61 = squeeze (mowgli)
                          [54.15, 0, -116.74]]  # -54.15, 0, -116.74 = side load (mowgli), -56.65, 0, -113.9 (musk)
    PopulateSimulation(sbm_data, perf_mesh, simulation, bottle_mass, translation_matrix)
