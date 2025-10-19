from django import template
register = template.Library()

@register.filter
def add_class(field, css):
    return field.as_widget(attrs={**(field.field.widget.attrs or {}), "class": css})

@register.filter
def mul(value, factor):
    try:
        return float(value) * float(factor)
    except (TypeError, ValueError):
        return 0
