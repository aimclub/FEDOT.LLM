from jinja2 import Environment, StrictUndefined


def jinja_render(template: str, *args, **kwargs):
    environment = Environment(undefined=StrictUndefined)
    return environment.from_string(template).render(*args, **kwargs)


def render(prompt, *args, **kwargs):
    system = prompt.get("system", None)
    if system:
        system = jinja_render(system, *args, **kwargs)
    user = jinja_render(prompt.user, *args, **kwargs)

    temperature = prompt.get("temperature", 0.2)
    frequency_penalty = prompt.get("frequency_penalty", 0.0)

    return user, system, temperature, frequency_penalty


def extract_code(response):
    return response.content.split("```python")[1].split("```")[0].strip()
