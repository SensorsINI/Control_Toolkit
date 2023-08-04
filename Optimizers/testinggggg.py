import tensorflow as tf
import time

size = (16, 15, 2)


def f1():
    i = tf.constant(0)
    final1 = tf.while_loop(lambda k: tf.less(k, 100), lambda k: (tf.add(k, 1),), [i])
    return final1


def f2():
    final2 = 0
    for i in range(100):
        final2 = i
    return final2


def main():
    averages = [0, 0]

    for i in range(100):
        begin1 = time.time()
        f1()
        end1 = time.time()
        delta_t1 = end1 - begin1
        averages[0] = (delta_t1 + averages[0] * i) / (i + 1)

        begin2 = time.time()
        f2()
        end2 = time.time()
        delta_t2 = end2 - begin2
        averages[1] = (delta_t2 + averages[1] * i) / (i + 1)

    check1 = f1()
    check2 = f2()

    time_ratio = averages[0]/averages[1]

    print(f'Average time for f1: {averages[0]}\n'
          f'Average time for f2: {averages[1]}\n'
          f'Do inputs coincide? {tf.reduce_all(tf.math.equal(check1, check2))}')

    if time_ratio < 0.90:
        print(f'f1 is faster {(1 / time_ratio):.2f} times')
    elif time_ratio > 1.10:
        print(f'f1 is slower {time_ratio:.2f} times')
    else:
        print('Both are more or less equal')


if __name__ == "__main__":
    main()
