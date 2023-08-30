FROM <python> as base

ENV TZ='Asia/Singapore'
ARG APPDIR=/app

WORKDIR $APPDIR

# copy application source
COPY ./src $APPDIR/src
COPY ./requirements.txt $APPDIR/requirements.txt
RUN python -m pip install -r $APPDIR/requirements.txt -i $PYPI_INTERNET_INDEX_URL --trusted-host $PIP_TRUSTED_HOST

# run the application
# ENTRYPOINT [ "/bin/sh" ]
# ENTRYPOINT [ "gradio src/app.py" ]