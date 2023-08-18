import pytest
from unittest import mock
import pandas as pd
import nfl_data_py as nfl

import nfl_wp_model.data as nfld
from support import (
    INVALID_PARAMETER,
    dummy_function
)

test_df = nfl.import_pbp_data([2022])


def test_data_class_assigns_import_function_output_to_data_attribute():
    # given
    data = nfld.Data(dummy_function, 1999, 2023, clean=False)

    # when
    expected = list(range(1999, 2023, 1)).reverse()
    actual = data.data
    # then
    assert expected == actual


def test_data_class_assigns_function_to_import_function_attribute():
    # given
    data = nfld.Data(dummy_function, 1999, 2023, clean=False)

    # then
    assert callable(data.import_function)


def test_data_class_assigns_start_season_parameter_to_start_season_attribute():
    # given
    expected = 1999
    # when
    data = nfld.Data(dummy_function, expected, 2023, clean=False)
    actual = data.start_season
    # then
    assert expected == actual


def test_data_class_assigns_end_season_parameter_to_end_season_attribute():
    # given
    expected = 2023
    # when
    data = nfld.Data(dummy_function, 1999, expected, clean=False)
    actual = data.end_season
    # then
    assert expected == actual


def test_data_class_attribute_seasons_range_returns_list_of_integers():
    # given
    data = nfld.Data(dummy_function, 1999, 2023, clean=False)
    # when
    actual = data.seasons_range
    # then
    assert isinstance(actual, list)
    assert isinstance(actual[0], int)


@mock.patch("nfl_wp_model.data.Data.get_data")
def test_data_class_get_data_called_during_init(get_data_mock):
    # given
    nfld.Data(dummy_function, 1999, 2023, clean=False)
    # then
    get_data_mock.assert_called_once()


@mock.patch("nfl_wp_model.data.Data.clean_data")
def test_data_class_clean_data_called_when_clean_true(clean_data_mock):
    # given
    nfld.Data(dummy_function, 1999, 2023, clean=True)
    # then
    clean_data_mock.assert_called_once()


@mock.patch("nfl_wp_model.data.Data.clean_data")
def test_data_class_clean_data_not_called_when_clean_false(clean_data_mock):
    # given
    nfld.Data(dummy_function, 1999, 2023, clean=False)
    # then
    clean_data_mock.assert_not_called()


@pytest.mark.parametrize("test_args", [
    (INVALID_PARAMETER, 1999, 2023, False),
    (dummy_function, INVALID_PARAMETER, 2023, False),
    (dummy_function, 1999, INVALID_PARAMETER, False),
    (dummy_function, 1999, 2023, INVALID_PARAMETER)
])
def test_data_class_init_raises_exception_for_invalid_type(test_args):
    # given
    test_function = test_args[0]
    test_starting_year = test_args[1]
    test_ending_year = test_args[2]
    test_clean = test_args[3]
    # then
    with pytest.raises(TypeError):
        nfld.Data(test_function, test_starting_year, test_ending_year, clean=test_clean)


def test_preprocess_class_assigns_input_to_raw_data_attribute():
    # given
    data = nfld.Preprocess(test_df)

    # when
    actual = data._raw_data
    # then
    assert isinstance(actual, pd.DataFrame)


def test_preprocess_class_assigns_input_to_data_attribute():
    # given
    data = nfld.Preprocess(test_df)

    # when
    actual = data.data
    # then
    assert isinstance(actual, pd.DataFrame)


def test_preprocess_class_selected_columns_attribute_is_list_of_strings():
    # given
    data = nfld.Preprocess(test_df)

    # when
    actual = data.selected_columns
    # then
    assert isinstance(actual, list)
    assert isinstance(actual[0], str)


@mock.patch("nfl_wp_model.data.Preprocess.preprocess_data")
def test_preprocess_class_preprocess_data_called_during_init(preprocess_data_mock):
    # given
    nfld.Preprocess(test_df)
    # then
    preprocess_data_mock.assert_called_once()


# @mock.patch("nfl_wp_model.data.Preprocess._add_label_column")
# @mock.patch("nfl_wp_model.data.Preprocess._set_column_type")
# @mock.patch("nfl_wp_model.data.Preprocess.preprocess_data")
# def test_preprocess_class_preprocess_data_calls_methods(mock_preprocess_data, mock_set_column_type, mock_add_label_column):
#     # given
#     nfld.Preprocess(test_df)
#     # then
#     mock_set_column_type.assert_called_once()
#     mock_add_label_column.assert_called_once()



# def test_select_columns():
#     assert False
#
#
# def test__set_column_types():
#     assert False
#
#
# def test__add_label_column():
#     assert False
#
#
# def test__add_home_column():
#     assert False
#
#
# def test__add_receive_2h_ko_column():
#     assert False
#
#
# def test__add_posteam_spread_column():
#     assert False
#
#
# def test__add_elapsed_share_column():
#     assert False
#
#
# def test__add_spread_time_column():
#     assert False
#
#
# def test__add_diff_time_ratio_column():
#     assert False
#
#
# def test__filter_rows_with_missing_values():
#     assert False
