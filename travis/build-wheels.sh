#!/bin/bash
set -e -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Build the SME Library
(cd /io/smelib/; sh /io/smelib/travis/build.sh)

# Copy files to the desired folder
cp -R /io/build/* /io/src/pysme/
ls /io/src/pysme/
ls /io/src/pysme/share/

# Compile wheels
ls /opt/python/
for PYBIN in /opt/python/cp3[6-9]*/bin; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/cp3[6-9]*/bin/; do
    "${PYBIN}/pip" install pysme-astro --no-index -f /io/wheelhouse
    "${PYBIN}/pytest"
done