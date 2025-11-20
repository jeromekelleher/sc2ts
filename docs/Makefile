# Need to set PYTHONPATH so that we pick up the local sc2ts
PYPATH=${PWD}/..
SC2TS_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import sc2ts; print(sc2ts.__version__.split("+")[0])')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for sc2ts version ${SC2TS_VERSION}
	sed -i s/__SC2TS_VERSION__/${SC2TS_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build

