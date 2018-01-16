# numerox makefile

PYTHON=python

help:
	@echo "Available tasks:"
	@echo "help    -->  This help page"
	@echo "test    -->  Run unit tests"
	@echo "flake8  -->  Check for pep8 errors"
	@echo "sdist   -->  Make source distribution"
	@echo "pypi    -->  Upload to pypi"
	@echo "coverage-->  html unit test coverage report"
	@echo "clean   -->  Remove all the build files for a fresh start"

test:
	${PYTHON} -c "import numerox; numerox.test()"

flake8:
	flake8 .

sdist: clean
	${PYTHON} setup.py sdist
	git status

pypi: clean
	${PYTHON} setup.py sdist upload -r pypi
	git status

coverage:
	rm -rf cover
	nosetests --with-coverage --cover-html --cover-package=numerox .
	firefox cover/index.html

clean:
	rm -f MANIFEST
	rm -rf build dist some_sums.egg-info
	find . -name \*.pyc -delete
	rm -rf build
