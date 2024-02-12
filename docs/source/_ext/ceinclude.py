import base64
import json

from docutils import nodes
from sphinx.directives.code import LiteralInclude, container_wrapper
from os import path

class CeIncludeDirective(LiteralInclude):
    """ LiteralInclude with a Compiler Explorer link"""

    def run(self):
        retnode = super().run()[0]

        _, filename = self.env.relfn2path(self.arguments[0])
        retnode = container_wrapper(self, retnode, path.basename(filename))

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
        ce_link = "https://godbolt.org/clientstate/" + encoded.decode()
        doc_link = f'<a href="{ce_link}" target="_blank">Edit on Compiler Explorer</a>'

        retnode += nodes.raw(rawsource = doc_link, text = doc_link, format="html")
        return [retnode]


def setup(app):
    app.add_directive("ceinclude", CeIncludeDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
