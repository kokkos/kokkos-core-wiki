import os
import shutil
import sys

try:
    project_path = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
    print(project_path)
    sys.path.append(project_path)
except Exception as e:
    print(f"Can not add project path to system path! Exiting!\nERROR: {e}")
    raise SystemExit(1)


class FileFinder:
    """Finds files with given extension in given directory and subdirectories as well. Returns a list of them."""
    def __init__(self, directory: str, file_extension: list):
        self.directory = os.path.abspath(directory)
        self.file_extension = file_extension
        self.files_list = None

    def get_files(self) -> list:
        """Returning found files."""
        if self.files_list is None:
            self.files_list = []
            for path, subdirs, files in os.walk(self.directory):
                for name in files:
                    if os.path.isfile(os.path.join(path, name)) and os.path.splitext(os.path.join(path, name))[-1] \
                            in self.file_extension:
                        self.files_list.append(os.path.join(path, name))
        return self.files_list


class HTMLButtonAdder:
    """Adds button for direct edition docs pages on GitHub."""
    def __init__(self, document_files: list, html_files: list, excluded_files: list, string_to_replace: str):
        self.document_files = sorted(document_files)
        self.html_files = sorted(html_files)
        self.excluded_files = excluded_files
        self.string_to_replace = string_to_replace
        self.__initial_checks()

    def __initial_checks(self):
        """Performs initial checks. Makes sure adding an edit button run smoothly."""
        print('==> Starting initial checks:')
        # Making sure the document_files has the same length as html_files
        if len(self.html_files) != len(self.document_files):
            raise AssertionError(f'Length of `html_files` list: {len(self.html_files)} is different than length of '
                                 f'`document_files` list: {len(self.document_files)}. Must be the same. '
                                 f'Check excluded files.')
        print(f'=> Found {len(self.html_files)} files')
        # Making sure that string to replace could be found in each html_file
        str_to_replace_list = []
        missing_files = []
        for file in self.html_files:
            with open(file) as html_file:
                html_str = html_file.read()
                if html_str.find(str_to_replace) != -1:
                    str_to_replace_list.append(file)
                else:
                    missing_files.append(file)
        if len(self.html_files) != len(str_to_replace_list):
            raise AssertionError(f'String to replace was not found in files: {missing_files}')
        print(f'=> Found {len(str_to_replace_list)} files with matching string to replace')
        print('--------------------------------------------------')

    def __overwrite_html(self, file_names: tuple, wiki_prefix: str, btn_file_name: str) -> None:
        """Overwriting html file with button addition."""
        # Setting relative path to the image
        generated_docs_dir = os.path.abspath(os.path.join(project_path, '../generated_docs'))
        up_dir = len(file_names[0].replace(generated_docs_dir, '').split(os.sep)) - 2
        up_dir_str = up_dir * '../'
        # Reading file, replacing string and overwriting
        with open(file_names[0], 'rt') as html_file:
            html_file_str = html_file.read()
        replaced_str = file_names[1].replace(project_path, wiki_prefix)
        str_to_put = f'<div>\n              <a href="{replaced_str}"><img src="{up_dir_str}_images/{btn_file_name}"' \
                     f' width="161" height="30"></a>\n            </div>'
        html_str_replace = html_file_str.replace(self.string_to_replace, str_to_put)
        with open(file_names[0], 'wt') as new_html_file:
            new_html_file.write(html_str_replace)
        print(f'=> Processing: {file_names[0]} done')

    def add_button(self, wiki_prefix: str, btn_file_name: str) -> None:
        """Loops over html files and overwrite them."""
        for num, file_names in enumerate(zip(self.html_files, self.document_files), 1):
            print(f'==> Processing pair {num}:\n=> {file_names[0]}\n=> {file_names[1]}')
            self.__overwrite_html(file_names=file_names, wiki_prefix=wiki_prefix, btn_file_name=btn_file_name)


if __name__ == "__main__":
    print('==================================================')
    print('==> Starting adding buttons to html files:')
    # Getting lists of documents and html files
    document_files = FileFinder(directory=project_path, file_extension=['.md', '.rst']).get_files()
    generated_docs_dir = os.path.join(project_path, '../generated_docs')
    html_files = FileFinder(directory=generated_docs_dir, file_extension=['.html']).get_files()
    # Excluded files (Files created by Sphinx, not to be overwritten with edit button)
    excluded_files = [os.path.abspath(os.path.join(generated_docs_dir, 'genindex.html')),
                      os.path.abspath(os.path.join(generated_docs_dir, 'search.html'))]
    # Final `html_files` list of files to add edit button to
    html_files = [html_file for html_file in html_files if html_file not in excluded_files]
    # String to replace
    str_to_replace = '<div class="icons">\n              \n            </div>'
    print(f'=> Replacing string: {str_to_replace}')
    # Wiki prefix pointing directly to GitHub
    wiki_prefix = 'https://github.com/kokkos/kokkos-core-wiki/edit/main/docs/source'
    print(f'=> Using prefix for Kokkos Wiki: {wiki_prefix}')
    # Button file name in source dir
    btn_file_name = 'edit_on_gh.png'
    print(f'=> Using: {btn_file_name} file for button')
    HTMLButtonAdder(document_files=document_files, html_files=html_files, excluded_files=excluded_files,
                    string_to_replace=str_to_replace).add_button(wiki_prefix=wiki_prefix, btn_file_name=btn_file_name)
    print('--------------------------------------------------')
    # Copying button file to `generated_docs`
    shutil.copy(f'{os.path.join(project_path, btn_file_name)}',
                os.path.abspath(os.path.join(project_path, f'../generated_docs/_images/{btn_file_name}')))
    print(f"=> Copied: {btn_file_name} to "
          f"{os.path.abspath(os.path.join(project_path, f'../generated_docs/_images/{btn_file_name}'))}")
    print('==================================================')
