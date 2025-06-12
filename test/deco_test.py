def make_pretty(func):
    def inner(*args, **kwargs):
        print("I got decorated")
        print(f"args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return inner


@make_pretty
def ordinary(conn, name, id):
    print(f"I am ordinary, {conn}, {name}, {id}")


ordinary(None, "T", 1)
