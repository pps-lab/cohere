

help:
	@echo 'Cohere Components'
	@echo '  make hyperparameter-explorer                        - launch the hyperparameter explorer'
	@echo '  make request-adapter-demo                           - run the request adapter demo'
	@echo '  make create-workloads                               - run the workload generator'
	@echo 'Cohere Evaluation Plots'
	@echo '  make plot-all                                       - recreate all plots from the paper'
	@echo '  make plot-subsampling                               - recreate the subsampling plot from the paper'
	@echo '  make plot-unlocking                                 - recreate the unlocking plot from the paper'
	@echo '  make plot-comparison                                - recreate the comparison plot from the paper'
	@echo 'Start Reproducing the Evaluation (using DoE-Suite on your remote machines e.g., AWS)'
	@echo '  make run-subsampling                               - start the REMOTE experiments underlying the subsampling plot'
	@echo '  make run-unlocking                                 - start the REMOTE experiments underlying the unlocking plot'
	@echo '  make run-comparison                                - start the REMOTE experiments underlying the comparison plot'
	@echo 'List Experiment Commands'
	@echo '  make cmd-subsampling                               - list all commands for the subsampling plot'
	@echo '  make cmd-unlocking                                 - list all commands for the unlocking plot'
	@echo '  make cmd-comparison                                - list all commands for the comparison plot'


##############################################################################
#   ___     _                   ___                                  _
#  / __|___| |_  ___ _ _ ___   / __|___ _ __  _ __  ___ _ _  ___ _ _| |_ ___
# | (__/ _ \ ' \/ -_) '_/ -_) | (__/ _ \ '  \| '_ \/ _ \ ' \/ -_) ' \  _(_-<
#  \___\___/_||_\___|_| \___|  \___\___/_|_|_| .__/\___/_||_\___|_||_\__/__/
#                                            |_|
##############################################################################

hyperparameter-explorer:
	@cd hyperparam-explorer && poetry install && poetry run python hyperparam_explorer/main.py


request-adapter-demo:
	@cd request-adapter && poetry install && poetry run python request_adapter/main.py


create-workloads:
	@cd workload-simulator && poetry install && poetry run python workload_simulator/main.py --output-dir $(out)/workloads --n-repetition 2

#############################

# Get the directory where the Makefile resides
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

out:=$(abspath $(MAKEFILE_DIR)/doe-suite-results-super-reproduce)


# results from the paper
sp_sub_default="sp_sub=1690826136"
sp_workloads_default="sp_unlock=1690743687 sp_workloads=1690805680"
sp_unlock_default="sp_unlock=1690743687 sp_workloads=1690805680"

##############################################################################
#    ___              _         ___ _     _
#   / __|_ _ ___ __ _| |_ ___  | _ \ |___| |_ ___
#  | (__| '_/ -_) _` |  _/ -_) |  _/ / _ \  _(_-<
#   \___|_| \___\__,_|\__\___| |_| |_\___/\__/__/
#
##############################################################################

plot-all: plot-subsampling plot-unlocking plot-comparison plot-perf

plot-subsampling:
	@echo "Building Subsampling Figure 5"
	@$(MAKE) run_super_etl default=$(sp_sub_default) config=sp_sub
	@echo "Finished creating subsampling plot: $(out)"
	@$(MAKE) latex-pdf-subsampling-fig5

plot-unlocking:
	@echo "Building Unlocking Figure 6"
	@$(MAKE) run_super_etl default=$(sp_unlock_default) config=sp_workloads pipelines="preprocess sim_round"
	@echo "Finished creating unlocking plot: $(out)"
	@$(MAKE) latex-pdf-unlocking-fig6

plot-comparison:
	@echo "Building Unlocking Figure 6"
	@$(MAKE) run_super_etl default=$(sp_workloads_default) config=sp_workloads pipelines="preprocess sim"
	@echo "Finished creating comparison plot: $(out)"
	@$(MAKE) latex-pdf-comparison-fig7

plot-perf: # TODO: could improve visualization
	echo "Building Performance Data"
	@$(MAKE) run_super_etl default=$(sp_workloads_default) config=sp_workloads pipelines="preprocess perf"
	echo "Finished creating perf plot: $(out)"

############################

# uses the doe-suite super etl functionality to build the plots based on the results
run_super_etl:
	@echo "Enter a result id:"; \
    read -p '  [default:$(default)] ' custom; \
	cd doe-suite && $(MAKE) etl-super config=$(config) out=$(out) custom-suite-id="$${custom:-$(default)}";

# Builds the figure as present in the paper
latex-pdf-%:
	cp doe-suite-config/super_etl/aux/$*.tex $(out)
	cd $(out) && latexmk -pdf && latexmk -c
	find $(out) -type f -name '*.fdb_latexmk' -delete
	find $(out) -type f -name '*.synctex.gz' -delete
	rm $(out)/$*.tex


###########################################################################
#   ___             ___                   _               _
#  | _ \_  _ _ _   | __|_ ___ __  ___ _ _(_)_ __  ___ _ _| |_ ___
#  |   / || | ' \  | _|\ \ / '_ \/ -_) '_| | '  \/ -_) ' \  _(_-<
#  |_|_\\_,_|_||_| |___/_\_\ .__/\___|_| |_|_|_|_\___|_||_\__/__/
#                          |_|
###########################################################################


suite_id?=new # alternative: last
RUN=cd doe-suite && $(MAKE) run suite=$(suite) id=$(suite_id)

run-subsampling:  # option to continue   (use make figure5-run suite_id=last to fetch the last run)
	echo "Starting Experiments for Subsampling (Figure 5)"
	$(eval suite := sp_sub)
	$(RUN)

run-unlocking: run-comparison
	echo "Starting Experiments for Unlocking Part 2/2 (Figure 6)"
	$(eval suite := sp_unlock)
	$(RUN)

run-comparison:
	echo "Starting Experiments for Unlocking Part 1/2 (Figure 6) Comparison (Figure 7)"
	$(eval suite := sp_workloads)
	$(RUN)


#################################################
#   ___                              _
#  / __|___ _ __  _ __  __ _ _ _  __| |___
# | (__/ _ \ '  \| '  \/ _` | ' \/ _` (_-<
#  \___\___/_|_|_|_|_|_\__,_|_||_\__,_/__/
#
#################################################


CMD=cd doe-suite && $(MAKE) design suite=$(suite)

cmd-subsampling:
	$(eval suite := sp_sub)
	$(CMD)

cmd-unlocking:
	$(eval suite := sp_unlock)
	$(CMD)
	$(eval suite := sp_workloads)
	$(CMD)

cmd-comparison:
	$(eval suite := sp_workloads)
	$(CMD)

############################
