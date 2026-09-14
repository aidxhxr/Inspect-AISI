@task
def theory_of_mind():
    return Task(
        dataset = json_dataset("theory_of_mind.jsonl"),
        solver = [
            system_message("system.txt"),
            prompt_template("prompt.txt"),
            generate(),
            self_critique(),
        ],
        scorer = model_graded_fact()
    )

# in the example above, I provided everything as a discrete block

@solver
def critique(system_prompt = "system.txt", user_prompt="user.txt"):
    return chain(
        system_message(system_prompt),
        prompt_template(user_prompt),
        generate(),
        self_critique(),
    )

@task
def theory_of_mind():
    return Task(
        dataset = json_dataset("theory_of_mind.jsonl"),
        solver = critique(),
        scorer = model_graded_fact(),
    )


# in the example below, I wrapped everything up in a single solver and provided it.