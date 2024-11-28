from jinja2 import Environment, StrictUndefined
from settings.config_loader import get_settings


def jinja_render(template: str, *args, **kwargs):
    environment = Environment(undefined=StrictUndefined)
    return environment.from_string(template).render(*args, **kwargs)


def render(prompt: str, *args, **kwargs):
    prompt_config = get_settings()[prompt]
    system = prompt_config.get('system', None)
    if system:
        system = jinja_render(system, *args, **kwargs)
    user = jinja_render(prompt_config.user, *args, **kwargs)

    temperature = prompt_config.get('temperature', 0.2)
    frequency_penalty = prompt_config.get('frequency_penalty', 0.0)

    return user, system, temperature, frequency_penalty


def extract_code(response):
    return response.content.split("```python")[1].split("```")[0].strip()
