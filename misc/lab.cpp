#include <vector>
#include <string>
#include <format>

int main(int argc, char** argv)
{
	std::vector<std::string> args(argv, argv + argc);
	if (argc >= 2)
	{
		std::string result = std::format("python ./lab{}/lab{}.py ", args[1], args[1]);
		for (int i = 2; i < args.size(); ++i)
		{
			result += args[i] + ' ';
		}
		system(result.c_str());
	}
}