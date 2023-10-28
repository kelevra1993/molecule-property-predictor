# Use the Python 3.10.10
FROM python:3.10.10

# Add Poetry to environment variables
ENV PATH="/root/.local/bin:$PATH"

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy Servier Folder To Our Docker Image
RUN mkdir -p /home/app/servier
COPY servier /home/app/servier

# Copy poetry.lock and pyproject.toml
COPY poetry.lock /home/app
COPY pyproject.toml /home/app
COPY README.md /home/app

# Set the working directory as app and install all dependencies
WORKDIR /home/app
RUN poetry install

# # Installation of Tensorflow and matplotlib is done out of poetry since has to be compatible with device
RUN poetry run pip install tensorflow==2.13.0


