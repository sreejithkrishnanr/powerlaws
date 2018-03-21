
def get_output_window_size(frequency):
    if frequency == 'D':
        return 59
    elif frequency == 'h':
        return 192
    elif frequency == '900s':
        return 192
    else:
        raise Exception('Unknown frequency %s' % (frequency, ))