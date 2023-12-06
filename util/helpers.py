def assert_success(function):
    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        assert res == True
    return wrapper
