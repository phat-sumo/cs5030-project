// validate.c - validates images for cs5030-project
// reads in two images, checks if they're the same, reports an error if not

#include <stdio.h>
#include <stdlib.h>

// help in case of bad arguments
void help(int argc, char* argv[], char error[]) {

	printf("%s invalid usage: ", argv[0]);

	for (int i = 0; i < argc; i++) {
		printf("%s ", argv[i]);
	}

	printf("\nerror: %s\n", error);

	printf("%s usage: %s FIRST_IMAGE SECOND_IMAGE\n", argv[0], argv[0]);
	
	exit(2);
}

int main(int argc, char* argv[]) {

	if (argc < 3) {
		help(argc, argv, "too few arguments");
	} else if (argc > 3) {
		help(argc, argv, "too many arguments");
	}



	return 0;
}
