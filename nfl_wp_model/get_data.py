import nfl_data_py as nfl


def get_pbp_data(starting_season, ending_season):
    data = nfl.import_pbp_data(range(starting_season, ending_season))
    nfl.clean_nfl_data(data)

    return data


def get_qbr_data(starting_season, ending_season):
    data = nfl.import_qbr(list(range(starting_season, ending_season)), frequency="weekly")
    nfl.clean_nfl_data(data)

    return data


def get_schedule_data(starting_season, ending_season):
    data = nfl.import_schedules(list(range(starting_season, ending_season)))
    nfl.clean_nfl_data(data)

    return data
