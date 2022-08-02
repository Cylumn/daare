"""
Converts USRP files to DigitalRF
Author: Anoush Khan
Date Created: Summer 2018
"""

import digital_rf as drf
import numpy as np
import os
import sys
import datetime
import dateutil.parser
import argparse


def parse_command_line(str_input=None):
    """This will parse through the command line arguments

    Function to go through the command line and if given a list of strings all
    also output a namespace object.

    Args:
        str_input (:obj:list): A list of strings from the command line. If none
            will read from commandline. Can just take command line inputs and do
            a split() on them.

    Returns:
        input_args(:obj:`Namespace`): An object holding the input arguments wrt the
            variables.
    """
    # if str_input is None:
    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="location of file to read from")
    parser.add_argument("-o", "--output", help="location of directory to output to")
    parser.add_argument("-a", "--antennas", help="number of antennas in data")
    parser.add_argument("-c", "--chunk", help="chunk size to read in bytes")
    parser.add_argument("-r", "--rate", help="sample rate in Hz")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Print status messages to stdout.")
    args = parser.parse_args()

    if args.verbose:
        print('Input: ' + args.input)
        print('Output: ' + args.output)
        print('Antennas: ' + args.antennas)
        print('Chunk Size: ' + args.chunk)
        print('Sample Rate in Hz: ' + args.rate)

    # ensure that arguments are passed
    try:
        sys.argv[1]
    except IndexError:
        print('Arguments required; use -h to see required arguments.')
        sys.exit()

    if str_input is None:
        return parser.parse_args()
    return parser.parse_args(str_input)


def make_dir(args):
    if not os.path.exists(args.output):
        if args.verbose:
            print('Output directory does not exist, making it now...')
        os.makedirs(args.output)

    # make directories to match number of antennas
    n_o_a = int(args.antennas)
    i = 0
    while i <= (n_o_a - 1):
        dir = (args.output + "/ant" + str(i))
        if not os.path.exists(dir):
            os.makedirs(dir)
        i += 1
    if args.verbose:
        print('Output directories created.')


def read_in_chunks(file_object, chunk_size):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def main_read(args):
    """
    """

    make_dir(args)

    # parameters
    # path or filename from argparse
    f_name = args.input
    # define data type (u2 = 2byte (16bit) unsigned integer)
    data_type = np.dtype([('r', np.int16), ('i', np.int16)])
    n_o_a = int(args.antennas)
    # data_type = ('u2,' * int(args.antennas))[:-1]
    # get filename
    path = os.path.splitext(args.input)[0]
    filename = path.split("/")[path.count('/')]

    if args.verbose:
        print('Data Type: ' + str(data_type))
        print('Path: ' + path)
        print('Filename: ' + filename)

    # convert start date and time into seconds since UNIX epoch
    UTstart = dateutil.parser.parse(filename.split("-")[1] + filename.split("-")[2])
    epoch = dateutil.parser.parse('1970, 1, 1')
    sec_since_epoch = int((UTstart - epoch).total_seconds())
    samples_since_epoch = sec_since_epoch * int(args.rate)

    if args.verbose:
        print('UTStart: ' + str(UTstart))
        print('Epoch: ' + str(epoch))
        print('Seconds since epoch: ' + str(sec_since_epoch))
        print('Samples since epoch: ' + str(samples_since_epoch))

    # confirm with user that all is fine before writing data
    # try:
    #     input("Press enter to convert data.")
    # except SyntaxError:
    #     pass

    # get conversion start time
    conversion_start = datetime.datetime.now()
    if args.verbose:
        print('Conversion start time: ' + str(conversion_start))

    # loop through number of antennas to write each to a channel
    b = 0
    while b <= (n_o_a - 1):
        # read data in chunks

        f = open(args.input, 'r')

        # define directory to write to
        dir = (args.output + "/ant" + str(b))

        # create digital_rf writer object
        writer = drf.DigitalRFWriter(
            dir, dtype=data_type,
            subdir_cadence_secs=3600, file_cadence_millisecs=1000,
            start_global_index=samples_since_epoch,
            sample_rate_numerator=(int(args.rate)), sample_rate_denominator=1,
            is_complex=False
        )

        # status messages if verbose flag enabled
        if args.verbose:
            print("Working directory: " + dir)
            print("Writing data for antenna " + str(b + 1))

        while True:
            d1 = np.fromfile(f, np.int16, int(args.chunk))

            d2write = np.zeros(d1.size // 2, dtype=data_type)
            d2write[:]['r'] = d1[::2]
            d2write[:]['i'] = d1[1::2]
            writer.rf_write(d2write)
            if d1.size < int(args.chunk):
                break
        # pass chunks to writer object to be written
        # for piece in read_in_chunks(f, int(args.chunk)):
        #     data = np.frombuffer(piece, type, count=-1)

        f.close()
        writer.close()

        b += 1

    # get conversion end time and calculate time delta
    conversion_end = datetime.datetime.now()
    if args.verbose:
        print('Conversion end time: ' + str(conversion_end))
    print('Time elapsed:' + str(conversion_end - conversion_start))


if __name__ == "__main__":
    args_commd = parse_command_line()
    main_read(args_commd)
