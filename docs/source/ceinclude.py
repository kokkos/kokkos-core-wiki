from docutils import nodes
from sphinx.directives.code import LiteralInclude, container_wrapper

class CeIncludeDirective(LiteralInclude):
    """ LiteralInclude with a Compiler Explorer link"""

    def run(self):
        retnode = super().run()[0]
        # paragraph_node = nodes.paragraph(text='Hello World!')
        retnode = container_wrapper(self, retnode, "https://godbolt.org/z/q9h339vob")
        return [retnode]

def setup(app):
    app.add_directive("ceinclude", CeIncludeDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }