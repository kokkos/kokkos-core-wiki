import base64
import json

from docutils import nodes
from sphinx.directives.code import LiteralInclude, container_wrapper

class CeIncludeDirective(LiteralInclude):
    """ LiteralInclude with a Compiler Explorer link"""

    def run(self):
        retnode = super().run()[0]
        _, filename = self.env.relfn2path(self.arguments[0])
        source = open(filename, "r").read()
        client_state = {
            "sessions": [
                {
                    "id": 1,
                    "language": "c++",
                    "source": source,
                    "compilers": [
                        {
                        "id": "g132",
                        "options": "-O3",
                        "libs": [{'name': 'kokkos', 'ver': '4100'}],
                        }
                    ],
                }
            ]
        }
        encoded = base64.urlsafe_b64encode(json.dumps(client_state).encode())
        retnode = container_wrapper(self, retnode, "https://godbolt.org/clientstate/" + encoded.decode())
        # breakpoint()
        return [retnode]

def setup(app):
    app.add_directive("ceinclude", CeIncludeDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }