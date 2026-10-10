from collections.abc import Callable
from typing import Any


class Registry[T]:
    """A registry that maps names to classes and creates their instances."""

    def __init__(self, registry_type: str) -> None:
        """Initializes an empty registry.

        Args:
            registry_type (str): Registry name used in error messages.
        """
        self._registry_type: str = registry_type
        self._registry_dict: dict[str, type[T]] = {}

    @staticmethod
    def _normalize(registration_name: str) -> str:
        """Normalizes a registration name.

        Strips leading and trailing whitespace and converts the name to lowercase.

        Args:
            registration_name (str): The name to normalize.

        Returns:
            str: The normalized name.

        Raises:
            ValueError: If the name is empty after stripping whitespace.
        """
        name: str = registration_name.strip().lower()
        if not name:
            raise ValueError("The registration name must not be empty")
        return name

    def register(self, registration_name: str) -> Callable[[type[T]], type[T]]:
        """Returns a decorator that registers a class under the given name.

        Args:
            registration_name (str): The name to register the class under.
                Case-insensitive; leading and trailing whitespace is ignored.

        Returns:
            Callable[[type[T]], type[T]]: A decorator that adds the class
                to the registry and returns it unchanged.

        Raises:
            ValueError: If the name is empty. Also raised when the decorator
                is applied if the name is already registered.
        """
        name: str = self._normalize(registration_name)

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry_dict:
                raise ValueError(f"Class with name '{registration_name}' is already "
                                 f"registered in '{self._registry_type}'")
            self._registry_dict[name] = cls
            return cls

        return decorator

    def create(self, registration_name: str, *args: Any, **kwargs: Any) -> T:
        """Creates an instance of the class registered under the given name.

        Args:
            registration_name (str): The name the class was registered under.
                Case-insensitive; leading and trailing whitespace is ignored.
            *args (Any): Positional arguments for the class constructor.
            **kwargs (Any): Keyword arguments for the class constructor.

        Returns:
            T: A new instance of the registered class.

        Raises:
            ValueError: If the name is empty.
            KeyError: If no class is registered under the given name.
        """
        name: str = self._normalize(registration_name)
        try:
            cls = self._registry_dict[name]
        except KeyError:
            available_names: str = ", ".join(sorted(self._registry_dict)) or "(none)"
            raise KeyError(f"Class with name '{registration_name}' is not registered in "
                           f"'{self._registry_type}'. Available names: {available_names}") from None
        return cls(*args, **kwargs)
