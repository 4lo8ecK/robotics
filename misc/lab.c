#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
	char buffer[64] = {0};
	if (argc == 2)
	{
		sprintf(buffer, "python ./lab%s/lab%s.py", argv[1], argv[1]);
		system(buffer);
	}
}