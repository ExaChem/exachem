#pragma once

namespace exachem::integrations::ipi {

/**
 * Open a socket connection to an i-PI server.
 *
 * @param psockfd Socket descriptor returned to caller.
 * @param inet    >0 for INET socket, 0 for UNIX domain socket.
 * @param port    Port number (INET mode only).
 * @param host    Hostname (INET) or socket name (UNIX).
 * @param sockets_prefix Prefix path for UNIX sockets.
 */
void open_socket(int* psockfd, int* inet, int* port, const char* host, const char* sockets_prefix);

/**
 * Write a buffer to the socket.
 *
 * @param psockfd Socket descriptor.
 * @param data    Buffer to send.
 * @param plen    Number of bytes to send.
 */
void writebuffer(int* psockfd, const char* data, int* plen);

/**
 * Read a buffer from the socket.
 *
 * @param psockfd Socket descriptor.
 * @param data    Destination buffer.
 * @param plen    Number of bytes to read.
 */
void readbuffer(int* psockfd, char* data, int* plen);

/**
 * Portable sleep function.
 *
 * @param seconds Sleep duration in seconds.
 */
void c_sleep(double seconds);

} // namespace exachem::integrations::ipi
