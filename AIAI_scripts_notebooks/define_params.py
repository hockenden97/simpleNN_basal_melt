# This python file defines the parameters for creating datasets and running the neural network
# This should avoid the need to copy and paste these into different notebooks/scripts and keep things consistent

# Define the different datasets used to explore the performance of the neural network 
def month_year_by_collection(this_collection, title = False):
    if this_collection == 'whole_dataset':
        title = 'The entire dataset is used as training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [1979, 
                      1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023]
    elif this_collection == 'no_mar_oct':
        title = 'March and October are used for testing'
        months_mask = [1,2,4,5,6,7,8,10,11,12]
        years_mask = [1979, 
                      1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023]
    elif this_collection == 'no_jun_dec':
        title = 'June and December are used for testing'
        months_mask = [1,2,3,4,5,7,8,9,10,11]
        years_mask = [1979, 
                      1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023]
    elif this_collection == 'up_to_2018':
        title = 'Simulations after 2018 are used for testing'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [1979, 
                      1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    elif this_collection == 'after_1984':
        title = 'Simulations before 1984 are used for testing'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                        1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023]
    elif this_collection == 'no_middle_5':
        title = 'Simulations from 1999-2003 are used for testing'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [1979, 
                      1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 
                                              2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023]
    elif this_collection == 'OPM026_whole_dataset':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 
                      2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
                      2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049,
                      2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059,
                      2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068]
    elif this_collection == 'OPM026_to2058':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 
                      2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
                      2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049,
                      2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058]
    elif this_collection == 'OPM026_to2048':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 
                      2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
                      2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048]
    elif this_collection == 'OPM026_to2038':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 
                      2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038]
    elif this_collection == 'OPM026_to2028':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
    elif this_collection == 'OPM026_to2018':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    elif this_collection == 'OPM026_to2008':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
    elif this_collection == 'OPM026_to1998':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988, 1989,
                      1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998]
    elif this_collection == 'OPM026_to1988':
        title = 'The whole dataset (1983-2068) is used for training'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                  1983, 1984, 1985, 1986, 1987, 1988]
    elif this_collection == 'OPM031_whole_dataset':
        title = 'The last 30 years of OPM031'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                                                      1999, 
                      2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                      2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
                      2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 
                      2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039,
                      2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049,
                      2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059,
                      2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069,
                      2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079,
                      2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089,
                      2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098]
    elif this_collection == 'OPM031_2069to2098':
        title = 'The last 30 years of OPM031'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                                                      2069,
                      2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079,
                      2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089,
                      2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098]
    elif this_collection == 'OPM031_2089to2098':
        title = 'The last 10 years of OPM031'
        months_mask = [1,2,3,4,5,6,7,8,9,10,11,12]
        years_mask = [                                                      2089,
                      2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098]
    else:
        print('I do not know this collection, help!')
    if title == True:
        return months_mask, years_mask, title
    else:
        return months_mask, years_mask

# Set list of desired variables for a particular experiment 
def var_list_by_exp_name(exp_name):
    if exp_name == 'all_vars':
        var_list = ['lat', 'lon', 'year', 'month', 'basins_NEMO', 'approx_area',
                   'distances_GL', 'distances_OO', 'distances_OC',
                   'temperature_prop', 'mean_T', 'std_T',
                   'salinity_prop', 'mean_S', 'std_S',
                   'corrected_isdraft', 'slope_is_lon', 'slope_is_lat', 'slope_is_across_front', 'slope_is_towards_front',
                   'bathymetry', 'slope_ba_lon', 'slope_ba_lat', 'slope_ba_across_front', 'slope_ba_towards_front',
                   'melt_m_ice_per_y']
    elif exp_name == 'reference':
        var_list = ['lat', 'lon', 'year', 'month', 'melt_m_ice_per_y']
    elif exp_name == 'slope_lat_lon':
        var_list =   ['distances_GL', 'distances_OO', 'distances_OC', 
                     'temperature_prop', 'salinity_prop', 'mean_T', 'mean_S', 'std_T', 'std_S',
                     'corrected_isdraft', 'slope_is_lon', 'slope_is_lat', 
                     'bathymetry', 'slope_ba_lon', 'slope_ba_lat',  
                     'melt_m_ice_per_y']
    elif exp_name == 'slope_front':
        var_list =   ['distances_GL', 'distances_OO', 'distances_OC',
                     'temperature_prop', 'mean_T', 'std_T',
                     'salinity_prop', 'mean_S', 'std_S',
                     'corrected_isdraft', 'slope_is_across_front', 'slope_is_towards_front',
                     'bathymetry', 'slope_ba_across_front', 'slope_ba_towards_front',
                     'melt_m_ice_per_y']
    return var_list

