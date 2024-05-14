
from typing import Callable
import attr, inspect

class Dog:
    tricks = []

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

class RequiredFieldValue:
    def __deepcopy__(self, memo):
        return self

    def __bool__(self):
        return False

    def __repr__(self):
        return "REQUIRED"


REQUIRED = RequiredFieldValue()
def _config_class_kwargs():
    return dict(init=False, kw_only=True, slots=True)


def _attr_field_from_signature_param(param: inspect.Parameter) -> attr.Attribute:
    default_value = param.default
    if default_value is inspect.Parameter.empty:
        default_value = REQUIRED
    return attr.field(default=default_value, type=param.annotation)

def config_for_function(fn: Callable) -> Dog:
    init_sig = inspect.signature(fn)

    config_attrs = {
        name: _attr_field_from_signature_param(param) for name, param in init_sig.parameters.items()
    }
    cls = attr.make_class(
        f"config_for_function({fn.__module__}.{fn.__qualname__})",
        bases=(Dog,),
        attrs=config_attrs,
        **_config_class_kwargs()
    )

    return cls(fn)



if __name__ == "__main__":
    def test():
        pass

    cls = config_for_function(test)
    print(issubclass(cls, Dog))
    print(cls.__name__)