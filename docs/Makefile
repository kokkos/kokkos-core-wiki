html:
	sphinx-build -b html ./source/ ./generated_docs/ -W -n --keep-going
	python3 ./source/edit_button_handler.py

clean:
	rm -rf ./generated_docs/
