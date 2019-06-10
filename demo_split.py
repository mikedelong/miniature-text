def fix(arg):
    result = list()
    skip = False
    for index, item in enumerate(arg):
        if item.endswith('-'):
            out_item = item[:-1]
            result.append(out_item + arg[index + 1])
            skip = True
        else:
            if skip:
                skip = False
            else:
                result.append(item)
    return result


if __name__ == '__main__':
    text = ['the', 'beginn-', 'ing', 'and']
    print(fix(text))
