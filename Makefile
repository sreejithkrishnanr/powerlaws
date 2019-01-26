.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = powerlaws
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

## Make Dataset
data: 
	mkdir -p 'data/interim'
	mkdir -p 'data/processed'
	
	make data_holidays
	make data_weather
	make data_split_train_data_by_freq
	make data_split_test_data_by_freq

## Make holiday dataset
data_holidays:
	$(PYTHON_INTERPRETER) src/data/make_holidays.py data/raw/holidays.csv data/raw/metadata.csv data/processed/holidays.hd5

## Make weather dataset
data_weather:
	$(PYTHON_INTERPRETER) src/data/make_weather.py data/raw/weather.csv data/processed/weather.hd5


## Split dataset according to forecast frequency
data_split_train_data_by_freq:
	$(PYTHON_INTERPRETER) src/data/make_split_data_by_freq.py data/raw/train.csv data/raw/submission_frequency.csv data/interim/

data_split_test_data_by_freq:
	$(PYTHON_INTERPRETER) src/data/make_split_data_by_freq.py data/raw/submission_format.csv data/raw/submission_frequency.csv data/interim/ --output_file_prefix=test


## Build features for per day forecast
features_per_day_train:
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim/train_86400000000000.hd5 data/processed/weather.hd5 data/raw/metadata.csv data/processed/holidays.hd5 data/processed/train_86400000000000.hd5  --frequency=D

features_per_day_test: features_per_day_train
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim/test_86400000000000.hd5 data/processed/weather.hd5 data/raw/metadata.csv data/processed/holidays.hd5 data/processed/test_86400000000000.hd5  --frequency=D --is_test_data=True --train_data_filepath=data/processed/train_86400000000000.hd5


## Build features for per hour forecast
features_per_hour_train:
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim/train_3600000000000.hd5 data/processed/weather.hd5 data/raw/metadata.csv data/processed/holidays.hd5 data/processed/train_3600000000000.hd5  --frequency=h

features_per_hour_test: features_per_hour_train
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim/test_3600000000000.hd5 data/processed/weather.hd5 data/raw/metadata.csv data/processed/holidays.hd5 data/processed/test_3600000000000.hd5  --frequency=h --is_test_data=True --train_data_filepath=data/processed/train_3600000000000.hd5


## Build features for per 15m forecast
features_per_15m_train:
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim/train_900000000000.hd5 data/processed/weather.hd5 data/raw/metadata.csv data/processed/holidays.hd5 data/processed/train_900000000000.hd5 --frequency=900s

features_per_15m_test: features_per_15m_train
	$(PYTHON_INTERPRETER) src/features/build_features.py data/interim/test_900000000000.hd5 data/processed/weather.hd5 data/raw/metadata.csv data/processed/holidays.hd5 data/processed/test_900000000000.hd5  --frequency=900s --is_test_data=True --train_data_filepath=data/processed/train_900000000000.hd5


## Build features for all forecasts
features_train: features_per_day_train features_per_hour_train features_per_15m_train
features_test: features_per_day_test features_per_hour_test features_per_15m_test
features: features_train features_test


## Build models
define make_build_args
	$(PYTHON_INTERPRETER) src/models/train_model.py $(1) --output_folder=$(2) --frequency=$(3) $(if $(4),--evaluate_only=$(4),) $(if $(5),--sites=$(5),) $(if $(6),--models=$(6),) $(if $(7),--verbose=$(7),)
endef

## Build model for per day forecast
model_build_per_day:
	$(call make_build_args,data/processed/train_86400000000000.hd5,models/freq1D,D,$(evaluate_only),$(sites),$(models),$(verbose))

## Build model for per hour forecast
model_build_per_hour:
	$(call make_build_args,data/processed/train_3600000000000.hd5,models/freq1h,h,$(evaluate_only),$(sites),$(models),$(verbose))

## Build model for per 15m forecast
model_build_per_15m:
	$(call make_build_args,data/processed/train_900000000000.hd5,models/freq900s,900s,$(evaluate_only),$(sites),$(models),$(verbose))

## Build all models
model_build: 
	make model_build_per_day models=gb,gb_log,gb_stat,gb_recursive
	make model_build_per_hour models=gb,gb_log,gb_stat,gb_recursive
	make model_build_per_15m models=gb,gb_log,gb_stat


## Predict for per day forecast
model_predict_per_day:
	$(PYTHON_INTERPRETER) src/models/predict_model.py data/processed/test_86400000000000.hd5 data/processed/train_86400000000000.hd5 predictions/test_86400000000000.hd5 --frequency=D --schema_filepath=models/freq1D/schema.json

## Predict for per hour forecast
model_predict_per_hour:
	$(PYTHON_INTERPRETER) src/models/predict_model.py data/processed/test_3600000000000.hd5 data/processed/train_3600000000000.hd5 predictions/test_3600000000000.hd5 --frequency=h --schema_filepath=models/freq1h/schema.json

## Predict for per 15m forecast
model_predict_per_15m:
	$(PYTHON_INTERPRETER) src/models/predict_model.py data/processed/test_900000000000.hd5 data/processed/train_900000000000.hd5 predictions/test_900000000000.hd5 --frequency=900s --schema_filepath=models/freq900s/schema.json

## Combine predictions
submission:
	$(PYTHON_INTERPRETER) src/data/make_submission.py data/raw/submission_format.csv predictions/test_86400000000000.hd5 predictions/test_3600000000000.hd5 predictions/test_900000000000.hd5 predictions/submission.csv

## Predict all models
model_predict: model_predict_per_day model_predict_per_hour model_predict_per_15m submission

## Build and predict
model: model_build model_predict

all:
	make requirements
	make data
	make features
	make model

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
