#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
	if (argc >= 2)
	{
		char buffer[1024] = {0};
		sprintf(buffer, "python ./lab%s/lab%s.py", argv[1], argv[1]);
		for (int i = 2; i < argc; ++i)
		{
			strcat(strcat(buffer, " "), argv[i]);
		}
		system(buffer);
	}
}