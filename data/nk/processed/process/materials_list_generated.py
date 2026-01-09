# Auto-generated material list
# Wavelength Range: 900-1700 nm

ALL_MATERIALS = [
    {'name': 'MgF2', 'category': 'LOW_INDEX', 'n': 1.4176, 'k': 0.00027, 'path': './MgF2.csv'},
    {'name': 'SiO2', 'category': 'LOW_INDEX', 'n': 1.4664, 'k': 0.0, 'path': './SiO2.csv'},
    {'name': 'ZnO', 'category': 'LOW_INDEX', 'n': 1.5745, 'k': 0.00205, 'path': './ZnO.csv'},
    {'name': 'MgO', 'category': 'MEDIUM_INDEX', 'n': 1.6961, 'k': 0.0, 'path': './MgO.csv'},
    {'name': 'Si3N4', 'category': 'MEDIUM_INDEX', 'n': 1.963, 'k': 0.0, 'path': './Si3N4.csv'},
    {'name': 'HfO2', 'category': 'HIGH_INDEX', 'n': 2.0586, 'k': 0.0, 'path': './HfO2.csv'},
    {'name': 'TiO2', 'category': 'HIGH_INDEX', 'n': 2.0644, 'k': 0.0, 'path': './TiO2.csv'},
    {'name': 'Ta2O5', 'category': 'HIGH_INDEX', 'n': 2.091, 'k': 0.0, 'path': './Ta2O5.csv'},
    {'name': 'AlN', 'category': 'HIGH_INDEX', 'n': 2.125, 'k': 0.0, 'path': './AlN.csv'},
    {'name': 'Nb2O5', 'category': 'HIGH_INDEX', 'n': 2.245, 'k': 0.0, 'path': './Nb2O5.csv'},
    {'name': 'ZnS', 'category': 'HIGH_INDEX', 'n': 2.2862, 'k': 1e-05, 'path': './ZnS.csv'},
    {'name': 'ZnSe', 'category': 'HIGH_INDEX', 'n': 2.4102, 'k': 0.02419, 'path': './ZnSe.csv'},
    {'name': 'Si', 'category': 'ULTRA_HIGH_INDEX', 'n': 3.4699, 'k': 0.0, 'path': './Si.csv'},
    {'name': 'a-Si', 'category': 'ULTRA_HIGH_INDEX', 'n': 3.5418, 'k': 0.00108, 'path': './a-Si.csv'},
    {'name': 'ITO', 'category': 'ABSORBER_LOSSY', 'n': 0.5821, 'k': 0.69269, 'path': './ITO.csv'},
    {'name': 'GaSb', 'category': 'ABSORBER_LOSSY', 'n': 4.0651, 'k': 0.16811, 'path': './GaSb.csv'},
    {'name': 'Ge', 'category': 'ABSORBER_LOSSY', 'n': 4.5957, 'k': 0.14646, 'path': './Ge.csv'},
    {'name': 'Ag', 'category': 'METAL', 'n': 0.1207, 'k': 9.59894, 'path': './Ag.csv'},
    {'name': 'Au', 'category': 'METAL', 'n': 0.1818, 'k': 9.2785, 'path': './Au.csv'},
    {'name': 'Al', 'category': 'METAL', 'n': 1.2014, 'k': 11.63079, 'path': './Al.csv'},
    {'name': 'Al2O3', 'category': 'METAL', 'n': 1.3265, 'k': 11.54337, 'path': './Al2O3.csv'},
    {'name': 'TiN', 'category': 'METAL', 'n': 2.5513, 'k': 4.39388, 'path': './TiN.csv'},
]

# Helper dictionary by category
MATERIALS_BY_CATEGORY = {
    'LOW_INDEX': ['MgF2', 'SiO2', 'ZnO'],
    'MEDIUM_INDEX': ['MgO', 'Si3N4'],
    'HIGH_INDEX': ['HfO2', 'TiO2', 'Ta2O5', 'AlN', 'Nb2O5', 'ZnS', 'ZnSe'],
    'ULTRA_HIGH_INDEX': ['Si', 'a-Si'],
    'ABSORBER_LOSSY': ['ITO', 'GaSb', 'Ge'],
    'METAL': ['Ag', 'Au', 'Al', 'Al2O3', 'TiN'],
}
