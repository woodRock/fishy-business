{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


.. automodule:: {{ fullname }}

   {% block modules %}
   {%- if modules %}
   .. rubric:: Modules

   .. autosummary::
      :toctree:
      :recursive:
      :template: module.rst
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :show-inheritance:
   {% endfor %}
   {% endif %}
   {%- endblock %}

   {%- block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {% endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   {% for item in functions %}
   .. autofunction:: {{ item }}
   {% endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {% endfor %}
   {% endif %}
   {%- endblock %}
   