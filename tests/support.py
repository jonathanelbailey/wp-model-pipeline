INVALID_PARAMETER = "invalid parameter"


def dummy_function(seasons):
    return seasons.reverse()

def generate_callable_methods(obj):
    return [x for x in dir(obj) if not x.startswith("__")]
