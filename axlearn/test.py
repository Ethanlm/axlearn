import cloudpickle
#import dill as cloudpickle


from dog import Dog, Color


def test():
    picked_file = "/tmp/test.cloudpickle"

    obj = Color.Lightgreen()

    print(f"ethan debug check isinstance current obj: {isinstance(obj, Color.Lightgreen)}")
    print(f"{id(obj.__class__)}")
    print(f"{id(Color.Lightgreen)}")

    obj2 = cloudpickle.loads(cloudpickle.dumps(obj))
    print(f"ethan debug check isinstance RELOADED obj: {isinstance(obj2, Color.Lightgreen)}")

    print(f"ethan debug reloaded , {id(obj2.__class__)==id(obj.__class__)}")

    print(f"{id(obj2.__class__)}")
    print(f"{id(Color.Lightgreen)}")

    with open(picked_file, "rb") as f:
        try:
            reload_obj = cloudpickle.load(f)
            print(f"ethan debug check isinstance RELOAD-LOCAL-prev-FILE obj: {isinstance(reload_obj, Color.Lightgreen)}")

            print(f"{id(reload_obj.__class__)}")
            print(f"{reload_obj.__class__}")
            print(f"{id(Color.Lightgreen)}")
            print(Color.Lightgreen)

        except Exception as e:
            print("ERROR: skipping: ", e)

    with open(picked_file, "wb") as f:
        #print(f"ethan debug write cloudpickle to {picked_file}")
        cloudpickle.dump(obj, f)

    with open(picked_file, "rb") as f:
        reload_obj = cloudpickle.load(f)
        print(f"ethan debug check isinstance RELOAD-LOCAL-FILE obj: {isinstance(reload_obj, Color.Lightgreen)}")
        print(f"{id(reload_obj.__class__)}")
        print(f"{id(Color.Lightgreen)}")

if __name__ == "__main__":
    test()